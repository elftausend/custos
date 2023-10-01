# Modules

`custos` ships combineable modules. Different selected modules result in different behaviour when executing operations.<br>
How is this possible? Module-related operations must be called in the custom operation. These module-related operations are defined via traits that can be implemented for any module.

A base(d) custom operation looks as follows:
(Element-wise addition)
```rust
// The definition of the operation. It will stay the same.
pub trait ElementWise<T, D: Device, S: Shape>: Device {
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
}

pub fn ew_add_slice<T: Add<Output = T> + Copy>(lhs: &[T], rhs: &[T], out: &mut [T]) {
    for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(out) {
        *out = *lhs + *rhs;
    }
}

// A base implementation for the `CPU` device. ("base" meaning only supporting the `Base` module)
// MainMemory: all devices that can access the RAM of a physical device. (e.g. device with unified memory )
impl<T: Add<Output = T> + Copy, D: MainMemory, S: Shape> ElementWise<T, D, S> for CPU {
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
        let mut out = self.buffer(lhs.len());
        ew_add_slice(lhs, rhs, &mut out);
        out
    }
}

```

This operations is going to support more and more modules as we go on.


## Cached

Automatically caches allocations.<br>
Use the `.retrieve(<size>, <parents>)` function to allow caching for an operation (also required for automatic differentiation).

Provided by the `Retrieve` and `Retriever` traits.

```diff
- let mut out = self.buffer(lhs.len());
+ let mut out = self.retrieve(lhs.len(), (lhs, rhs));
```

As the operation now must support arbitrarily combined modules, a new generic, usually called `Mods` must be added. 

```diff
- impl<T: Add<Output = T> + Copy, D: MainMemory, S: Shape> ElementWise<T, D, S> for CPU {
+ impl<T: Add<Output = T> + Copy, D: MainMemory, S: Shape, Mods: Retrieve<Self, T>>
+    ElementWise<T, D, S> for CPU<Mods>
```

The caching (and autograd) system use `#[track_caller]` to determine the required cache entry.

```diff
+ #[track_caller]
```

```rust
impl<T: Add<Output = T> + Copy, D: MainMemory, S: Shape, Mods: Retrieve<Self, T>>
    ElementWise<T, D, S> for CPU<Mods>
{
    #[track_caller]
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        ew_add_slice(lhs, rhs, &mut out);
        out
    }
}
```


## Lazy

Adds lazy exection support.<br>
Provided by the `AddOperation` trait.

```diff
- ew_add_slice(lhs, rhs, &mut out);
+ self.add_op(&mut out, |out| ew_add_slice(lhs, rhs, out));
```

```rust
impl<T, D, S, Mods> ElementWise<T, D, S> for CPU<Mods>
// moved trait bounds to where clause
where
    T: Add<Output = T> + Copy,
    D: MainMemory,
    S: Shape,
    Mods: Retrieve<Self, T> + AddOperation<T, Self>,
{
    #[track_caller]
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        self.add_op(&mut out, |out| ew_add_slice(lhs, rhs, out));
        out
    }
}
```

## Autograd

This module adds support for automatic differentiation.<br>
To add a gradient function, do the following:

```rust
// the grad function. (partial derivative for `a + b` with respect to `a` and `b` and chain rule) 
pub fn add_ew_grad_slice<T>(lhs_grad: &mut [T], rhs_grad: &mut [T], out: &[T])
where
    T: Copy + AddAssign + Mul<Output = T>,
{
    for ((lhs_grad, rhs_grad), out) in lhs_grad.iter_mut().zip(rhs_grad).zip(out) {
        *lhs_grad += *out;
        *rhs_grad += *out;
    }
}
```

```diff
+ #[cfg(feature = "autograd")]
+ {
+     let ids = (lhs.id(), rhs.id(), out.id()); // trackable buffer ids
+     self.add_grad_fn(move |grads| {
+         let (_lhs, _rhs, lhs_grad, rhs_grad, out_grad) =
+             grads.get_triple::<T, S, Self>(ids); // retrieve buffers from gradient cache
+         add_ew_grad_slice(lhs_grad, rhs_grad, out_grad) // execute grad function
+     });
+ }
```
After adding some trait bounds, primarily `MayTapeActions`:

```rust

impl<T, D, S, Mods> ElementWise<T, D, S> for CPU<Mods>
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + 'static,
    D: MainMemory,
    S: Shape,
    Mods: Retrieve<Self, T> + MayTapeActions + 'static,
{
    #[track_caller]
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));

        #[cfg(feature = "autograd")]
        {
            let ids = (lhs.id(), rhs.id(), out.id()); // trackable buffer ids
            self.add_grad_fn(move |grads| {
                let (_lhs, _rhs, lhs_grad, rhs_grad, out_grad) =
                    grads.get_triple::<T, S, Self>(ids); // retrieve buffers from gradient cache
                add_ew_grad_slice(lhs_grad, rhs_grad, out_grad) // execute grad function
            });
        }

        self.add_op(&mut out, |out| ew_add_slice(lhs, rhs, out));
        out
    }
}
```

## Fork

Decides whether the CPU or GPU is faster for an operation. It then uses the faster device for following computations. This is useful for devices with unified memory.<br>
The trait is `UseGpuOrCpu`.

Now, the `OpenCL` device is used.

```rust
// the operation that is executed
pub fn try_add_ew_cl<T, Mods>(
    device: &OpenCL<Mods>,
    lhs: &CLPtr<T>,
    rhs: &CLPtr<T>,
    out: &mut CLPtr<T>,
) -> custos::Result<()>
where
    T: CDatatype + Default,
{
    let src = format!(
        "
        __kernel void add_ew(__global const {ty}* lhs, __global const {ty}* rhs, __global {ty}* out) {{
            size_t id = get_global_id(0);
            out[id] = lhs[id] + rhs[id];
        }}
    ",
        ty = T::C_DTYPE_STR,
    );

    device.launch_kernel(
        &src,
        [((lhs.len + 32) / 32) * 32, 0, 0],
        Some([32, 0, 0]),
        &[lhs, rhs, out],
    )
}
```

```diff
+ #[cfg(unified_cl)]
+ {
+     let cpu_out = unsafe { &mut *(out as *mut Buffer<_, _, _>) };
+     self.use_cpu_or_gpu(
+         (file!(), line!(), column!()).into(),
+         &[lhs.len()],
+         || add_ew_slice(lhs, rhs, cpu_out),
+         || try_add_ew_cl(self, &lhs.data, &rhs.data, &mut out.data).unwrap(),
+     );
+ }
+ #[cfg(not(unified_cl))]
+ try_add_ew_cl(self, lhs, rhs, out).unwrap();
```

```rust
#[cfg(feature = "opencl")]
impl<T, S, Mods> ElementWise<T, Self, S> for custos::OpenCL<Mods>
where
    T: Add<Output = T> + Copy + CDatatype + Default,
    S: Shape,
    Mods: Retrieve<Self, T> + AddOperation<T, Self> + UseGpuOrCpu,
{
    fn add(&self, lhs: &Buffer<T, Self, S>, rhs: &Buffer<T, Self, S>) -> Buffer<T, Self, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));

        self.add_op(&mut out, |out| {
            #[cfg(unified_cl)]
            {
                let cpu_out = unsafe { &mut *(out as *mut Buffer<_, _, _>) };
                self.use_cpu_or_gpu(
                    (file!(), line!(), column!()).into(),
                    &[lhs.len()],
                    || add_ew_slice(lhs, rhs, cpu_out),
                    || try_add_ew_cl(self, &lhs.data, &rhs.data, &mut out.data).unwrap(),
                );
            }
            #[cfg(not(unified_cl))]
            try_add_ew_cl(self, lhs, rhs, out).unwrap();
        });

        out
    }
}
```

Try it with:
```rust        
let device = OpenCL::<Fork<Lazy<Cached<Base>>>>::new(0).unwrap();
let lhs = device.buffer([1, 2, 3, 4, 5]);
let rhs = device.buffer([1, 2, 3, 4, 5]);

let out = device.add(&lhs, &rhs);
```

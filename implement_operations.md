# custos - implement operations for all compute devices (CPU, OpenCL, CUDA).

As code in .md files is not compiled directly, minor compilation issues may appear in the snippets below. 
Hence, you can find the entire source code in the examples folder, or just click [here](https://github.com/elftausend/custos/blob/main/examples/implement_operations.rs).

## Make operations invocable for all devices

Implementing a new operation happens in a few steps. <br>

The first step is to define a trait. This is the operation<br>
This trait is then implemented for all devices.<br>

You can use your own type as parameters also. It just needs to encapsulate a ```Buffer```.

```rust
/// AddBuf will be implemented for all compute devices.
pub trait AddBuf<T>: Sized {
    /// This operation perfoms element-wise addition.
    fn add(&self, lhs: &Buffer<T, Self>, rhs: &Buffer<T, Self>) -> Buffer<T, Self>;
    // ... you can add more operations if you want to do that.
}
```

Afterwards, implement this trait for every device.<br>
However, it is not mandatory to implement this trait for every device, but we will come to this again a bit later.

```rust
// Host CPU implementation
impl<T> AddBuf<T> for CPU
where
    T: Copy + std::ops::Add<Output = T>, // you can use the custos::Number trait. 
                                         // This trait is implemented for all number types (usize, i16,      f32, ...)
{
    fn add(&self, lhs: &Buffer<T, CPU>, rhs: &Buffer<T, CPU>) -> Buffer<T, CPU> {
        let len = std::cmp::min(lhs.len, rhs.len);

        // this returns a previously allocated buffer.
        // You can deactivate the caching behaviour by adding the "realloc" feature
        // to the custos feature list in the Cargo.toml.
        let mut out = Cache::get(self, len, [lhs.node.idx, rhs.node.idx]);

        // By default, the Buffer dereferences to a slice.
        // Therefore, standard indexing can be used.
        // You can pass a CPU Buffer to a function that takes a slice as a parameter, too.
        // However, the Buffer must be created via a CPU.
        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }
        out
    }
}

// OpenCL implementation
impl<T> AddBuf<T> for OpenCL
where
    T: CDatatype, // the custos::CDatatype trait is used to
{
    // get the OpenCL C type string for creating generic OpenCL kernels.
    fn add(&self, lhs: &Buffer<T, OpenCL>, rhs: &Buffer<T, OpenCL>) -> Buffer<T, OpenCL> {
        // generic OpenCL kernel
        let src = format!("
            __kernel void add(__global const {datatype}* lhs, __global const {datatype}* rhs, __global {datatype}* out) 
            {{
                size_t id = get_global_id(0);
                out[id] = lhs[id] + rhs[id];
            }}
        ", datatype=T::as_c_type_str());

        let len = std::cmp::min(lhs.len, rhs.len);
        let out = Cache::get::<T, OpenCL>(self, len, [lhs.node.idx, rhs.node.idx]);

        // In the background, the kernel is compiled once. After that, it will be reused for every iteration.
        // The cached kernels are released (or freed) when the underlying CLDevice is dropped.
        // The arguments are specified with a slice of buffers and/or numbers.
        enqueue_kernel(self, &src, [len, 0, 0], None, &[&lhs, &rhs, &out]).unwrap();
        out
    }
}

#[cfg(feature = "cuda")]
// CUDA Implementation
impl<T: CDatatype> AddBuf<T> for CUDA {
    fn add(&self, lhs: &Buffer<T, CUDA>, rhs: &Buffer<T, CUDA>) -> Buffer<T, CUDA> {
        // generic CUDA kernel
        let src = format!(
            r#"extern "C" __global__ void add({datatype}* lhs, {datatype}* rhs, {datatype}* out, int numElements)
                {{
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if (idx < numElements) {{
                        out[idx] = lhs[idx] + rhs[idx];
                    }}
                    
                }}
        "#,
            datatype = T::as_c_type_str()
        );

        let len = std::cmp::min(lhs.len, rhs.len);
        let out = Cache::get::<T, CUDA>(self, len, (lhs.node.idx, rhs.node.idx));

        // The kernel is compiled once with nvrtc and is cached too.
        // The arguments are specified with a vector of buffers and/or numbers.
        launch_kernel1d(len, self, &src, "add", &[lhs, rhs, &out, &len]).unwrap();
        out
    }
}
```
Now, the operation is accessible via the devices.<br>
Let's try it out with a ```CPU``` device:

```rust
let device = CPU::new();

let lhs = Buffer::from((&device, [1, 3, 5, 3, 2, 6]));
let rhs = Buffer::from((&device, [-1, -12, -6, 3, 2, -1]));

let out = device.add(&lhs, &rhs);
assert_eq!(out.read(), vec![0, -9, -1, 6, 4, 5]); // to read a CPU Buffer, you can also call .as_slice() on it.
```

If you want to use another device, just update the device declaration.

```rust
fn main() -> custos::Result<()> {
    let device = CLDevice::new(0)?; 
    // or:
    let device = CudaDevice::new(0)?;
    Ok(())
}
```

## Make operations invocable on ```Buffer``` or custom structs

Now, we have implemented a custom operation for all compute devices. 
However, you may have spot something.

```rust
let out = device.add(&lhs, &rhs);
assert_eq!(out.read(), vec![0, -9, -1, 6, 4, 5]);
```

If you take a closer look at both statements, you may notice that the ```read()``` function is invoked on a buffer without specifying any device, even though reading a buffer is device specific.
We can implement this for ```AddBuf``` as well.<br>
To get this to work, a new trait must be created. If the operation should be used on a struct created in your current crate, you can omit this step.

```rust
pub trait AddOp<'a, T, D> {
    fn add(&self, rhs: &Buffer<'a, T, D>) -> Buffer<'a, T, D>;
}
```

This trait is then implemented for ```Buffer``` (or any other type).

```rust
impl<'a, T: CDatatype, D: AddBuf<T>> AddOp<'a, T, D> for Buffer<'a, T, D> {
    #[inline]
    fn add(&self, rhs: &Buffer<'a, T, D>) -> Buffer<'a, T, D> {
        self.device().add(self, rhs)
    }
}
```

If you have defined your own struct that encapsulates a ```Buffer```, you can do the following:

```rust
pub struct OwnStruct<'a, T, D> {
    buf: Buffer<'a, T, D>,
}

impl<'a, T, D> OwnStruct<'a, T, D> {
    #[allow(dead_code)]
    // consider using operator overloading for your own type
    #[inline]
    fn add(&self, rhs: &OwnStruct<T, D>) -> Buffer<T, D>
    where
        T: CDatatype,
        D: AddBuf<T>
    {
        self.buf.device().add(&self.buf, &rhs.buf)
    }

    // general context
    /*#[inline]
    fn operation(&self, rhs: &OwnStruct<T>, other_arg: &T) -> OwnStruct<T> {
        get_device!(self.buf.device, OperationTrait<T>).operation(self, rhs, other_arg)
    }*/

    // ...
}
```

Without specifying a device:

```rust
let out = lhs.add(&rhs);
assert_eq!(out.read(), vec![0, -9, -1, 6, 4, 5]);
```
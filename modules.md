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

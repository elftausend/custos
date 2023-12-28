![custos logo](assets/custos.png)

<hr/>

[![Crates.io version](https://img.shields.io/crates/v/custos.svg)](https://crates.io/crates/custos)
[![Docs](https://docs.rs/custos/badge.svg?version=0.7.0)](https://docs.rs/custos/0.7.0/custos/)
[![Rust](https://github.com/elftausend/custos/actions/workflows/rust.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/rust.yml)
[![GPU](https://github.com/elftausend/custos/actions/workflows/gpu.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/gpu.yml)
[![rust-clippy](https://github.com/elftausend/custos/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/rust-clippy.yml)
[![Android NNAPI](https://github.com/elftausend/custos/actions/workflows/android.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/android.yml)

A minimal, extensible OpenCL, Vulkan (with WGSL), CUDA, NNAPI (Android) and host CPU array manipulation engine / framework written in Rust. 
This crate provides tools for executing custom array and automatic differentiation operations.<br>


## Installation

Add "custos" as a dependency:
```toml
[dependencies]
custos = "0.7.0"

# to disable the default features (cpu, cuda, opencl, static-api, blas, macro) and use an own set of features:
#custos = {version = "0.7.0", default-features=false, features=["opencl", "blas"]}
```

### Available features: 

custos ships combineable modules. Different selected modules result in different behaviour when executing operations.
New modules can be added in user code.
```rust
use custos::prelude::*; 
// Autograd, Base = Modules
let device = CPU::<Autograd<Base>>::new();
```
To make specific modules useable for building a device, activate the corresponding features:

Feature | Module | Description
--- | --- | ---
on by default | `Base` | Default behaviour.
autograd | `Autograd` | Enables running automatic differentiation.
cached | `Cached` | Reuses allocations on demand.
fork | `Fork` | Decides whether the CPU or GPU is faster for an operation. It then uses the faster device for following computations. (unified memory devices)
lazy | `Lazy` | Lazy execution of operations. Enables support for CUDA graphs. However, allocations are not lazy at the moment.
graph | `Graph` | Adds a memory usage optimizeable graph.

Usage of these modules when writing custom operations: [`modules.md`](modules.md)

If an operations wants to be affected by a module, specific custos code must be called in that operation.

To make specific devices useable, activate the corresponding features:

Feature | Device | Notes
--- | --- | ---
cpu | `CPU` | Uses heap allocations.
stack | `Stack` | Useable in `no-std` environments as it uses stack allocated `Buffer`s. Practically only supports the `Base` module.
opencl | `OpenCL` | Currently the only device that supports automatic unified memory mapping. 
cuda | `CUDA` |
vulkan | `Vulkan` | Shaders are written in WGSL.
nnapi | `NnapiDevice` | `Lazy` module is mandatory.

Remaining features: 

Feature | Description
--- | --- 
no-std | For no std environments, activates `stack` feature.
static-api | Enables the creation of `Buffer`s without providing a device.
macro | Reexport of [custos-macro]
blas | Adds gemm functions of the system's (selected) BLAS library.

<!-- Feature | Description -->
<!-- --- | --- -->
<!-- opt-cache | Makes the 'cache graph' optimizeable, lowering the memory footprint. -->

[custos-macro]: https://github.com/elftausend/custos-macro

## [Examples]


[examples]: https://github.com/elftausend/custos/tree/main/examples
[unary]: https://github.com/elftausend/custos/blob/main/src/unary.rs

Implement an operation for `CPU`:<br>
- If you want to implement your own operations for all compute devices, consider looking here: [implement_operations.md](implement_operations.md)<br>
or to see it at a larger scale, look here [`custos-math`](https://github.com/elftausend/custos-math) or here [`sliced`](https://github.com/elftausend/sliced) (for automatic diff examples).

This operation is only affected by the `Cached` module (and partially `Autograd`).

```rust
use std::ops::Mul;
use custos::prelude::*;

pub trait MulBuf<T, S: Shape = (), D: Device = Self>: Sized + Device {
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
}

impl<Mods, T, S, D> MulBuf<T, S, D> for CPU<Mods>
where
    Mods: Retrieve<Self, T>,
    T: Mul<Output = T> + Copy + 'static,
    S: Shape,
    D: Device,
    D::Data<T, S>: core::ops::Deref<Target = [T]>
{
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));

        for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(&mut out) {
            *out = *lhs * *rhs;
        }

        out
    }
}
```

A lot more usage examples can be found in the [tests] and [examples] folders.
(Or in the [unary] operation file, [custos-math](https://github.com/elftausend/custos-math) and [`sliced`](https://github.com/elftausend/sliced))

[tests]: https://github.com/elftausend/custos/tree/main/tests
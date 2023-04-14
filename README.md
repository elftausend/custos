![custos logo](assets/custos.png)

<hr/>

[![Crates.io version](https://img.shields.io/crates/v/custos.svg)](https://crates.io/crates/custos)
[![Docs](https://docs.rs/custos/badge.svg?version=0.6.3)](https://docs.rs/custos/0.6.3/custos/)
[![Rust](https://github.com/elftausend/custos/actions/workflows/rust.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/rust.yml)
[![GPU](https://github.com/elftausend/custos/actions/workflows/gpu.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/gpu.yml)
[![rust-clippy](https://github.com/elftausend/custos/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/rust-clippy.yml)

A minimal OpenCL, WGPU, CUDA and host CPU array manipulation engine / framework written in Rust.
This crate provides the tools for executing custom array and automatic differentiation operations with the CPU, as well as with CUDA, WGPU and OpenCL devices.<br>
This guide demonstrates how operations can be implemented for the compute devices: [implement_operations.md](implement_operations.md)<br>
or to see it at a larger scale, look here [`custos-math`](https://github.com/elftausend/custos-math) or here [`sliced`](https://github.com/elftausend/sliced) (for automatic diff examples).


## Installation

Add "custos" as a dependency:
```toml
[dependencies]
custos = "0.6.3"

# to disable the default features (cpu, cuda, opencl, static-api, blas, macro) and use an own set of features:
#custos = {version = "0.6.3", default-features=false, features=["opencl", "blas"]}
```

### Available features: 

Feature | Description
--- | ---
cpu | Adds the `CPU` device
stack | Adds the `Stack` device, enables stack allocated `Buffer`s
opencl | Adds OpenCL features. (name of the device: `OpenCL`)
cuda | Adds CUDA features. (name of the device: `CUDA`)
wgpu | Adds WGPU features. (name of the device: `WGPU`)
no-std | For no std environments, activates `stack` feature.
static-api | Enables the creation of `Buffer`s without providing a device.
blas | Adds gemm functions from the system's (selected) BLAS library.
opt-cache | Makes the 'cache graph' optimizeable, lowering the memory footprint.
macro | Reexport of [custos-macro]
realloc | Disables allocation caching for all devices.
autograd | Adds automatic differentiation features (currently only in the git repository).

[custos-macro]: https://github.com/elftausend/custos-macro

## [Examples]

custos only implements four `Buffer` operations. These would be the `write`, `read`, `copy_slice` and `clear` operations, however, there are also [unary] (device only) operations.<br>
On the other hand, [custos-math] implements a lot more operations, including Matrix operations for a custom Matrix struct.<br>

[examples]: https://github.com/elftausend/custos/tree/main/examples
[unary]: https://github.com/elftausend/custos/blob/main/src/unary.rs

Implement an operation for `CPU`:
If you want to implement your own operations for all compute devices, consider looking here: [implement_operations.md](implement_operations.md)

```rust
use std::ops::Mul;
use custos::prelude::*;

pub trait MulBuf<T, S: Shape = (), D: Device = Self>: Sized + Device {
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
}

impl<T, S, D> MulBuf<T, S, D> for CPU
where
    T: Mul<Output = T> + Copy,
    S: Shape,
    D: MainMemory,
{
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, CPU, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));

        for ((lhs, rhs), out) in lhs.iter().zip(&*rhs).zip(&mut out) {
            *out = *lhs * *rhs;
        }

        out
    }
}
```

A lot more usage examples can be found in the [tests] and [examples] folders.
(Or in the [unary] operation file, [custos-math](https://github.com/elftausend/custos-math) and [`sliced`](https://github.com/elftausend/sliced))

[tests]: https://github.com/elftausend/custos/tree/main/tests
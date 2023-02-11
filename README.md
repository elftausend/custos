![custos logo](assets/custos.png)

<hr/>

[![Crates.io version](https://img.shields.io/crates/v/custos.svg)](https://crates.io/crates/custos)
[![Docs](https://docs.rs/custos/badge.svg?version=0.6.3)](https://docs.rs/custos/0.6.3/custos/)
[![Rust](https://github.com/elftausend/custos/actions/workflows/rust.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/rust.yml)
[![GPU](https://github.com/elftausend/custos/actions/workflows/gpu.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/gpu.yml)
[![rust-clippy](https://github.com/elftausend/custos/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/rust-clippy.yml)

A minimal OpenCL, WGPU, CUDA and host CPU array manipulation engine / framework written in Rust.
This crate provides the tools for executing custom array operations with the CPU, as well as with CUDA, WGPU and OpenCL devices.<br>
This guide demonstrates how operations can be implemented for the compute devices: [implement_operations.md](implement_operations.md)<br>
or to see it at a larger scale, look here [custos-math] or here [sliced].

[custos-math]: https://github.com/elftausend/custos-math
[sliced]: https://github.com/elftausend/sliced

## Installation

Add "custos" as a dependency:
```toml
[dependencies]
custos = "0.6.3"

# to disable the default features (cpu, cuda, opencl, static-api, blas, macro) and use an own set of features:
#custos = {version = "0.6.3", default-features=false, features=["opencl", "blas"]}
```

Available features: 
- devices
    - "cpu" ... adds `CPU` device
    - "stack" ... adds `Stack` device and enables stack allocated `Buffer`
    - "opencl" ... adds OpenCL features. (name of the device: `OpenCL`)
    - "cuda" ... adds CUDA features. (name of the device: `CUDA`)
    - "wgpu" ... adds WGPU features.(name of the device: `WGPU`)

- "no-std" ... for no std environments, activates "stack" feature
- "static-api" ... enables the creation of `Buffer` without providing any device.
- "blas" ... adds gemm functions from your selected BLAS library
- "opt-cache" ... makes the 'cache graph' optimizeable
- "macro" ... reexport of [custos-macro]
- "realloc" ... disables caching for all devices

[custos-macro]: https://github.com/elftausend/custos-macro

## [Examples]

These examples show how to use the implemented operations. <br>
custos only implements four `Buffer` operations. These would be the `write`, `read`, `copy_slice` and `clear` operations.<br>
On the other hand, [custos-math] implements a lot more operations, including Matrix operations for a custom Matrix struct.<br>
If you want to implement your own operations for all compute devices, look here: [implement_operations.md](implement_operations.md)

[examples]: https://github.com/elftausend/custos/tree/main/examples

Using the host CPU as the compute device:

[cpu_readme.rs]

[cpu_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cpu_readme.rs
```rust
use custos::{Buffer, ClearBuf, Read, CPU};

fn main() {
    let device = CPU::new();
    let mut a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    // specify device for operation
    device.clear(&mut a);
    assert_eq!(device.read(&a), [0; 6]);

    let device = CPU::new();

    let mut a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    // no need to specify the device
    a.clear();
    assert_eq!(a.read(), vec![0; 6]);
}
```

Using an `OpenCL` device as the compute device:

[cl_readme.rs]

[cl_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cl_readme.rs
```rust
use custos::{Buffer, OpenCL};

fn main() -> custos::Result<()> {
    let device = OpenCL::new(0)?;

    let mut a = Buffer::from((&device, [5, 3, 2, 4, 6, 2]));
    a.clear();

    assert_eq!(a.read(), [0; 6]);
    Ok(())
}

```

Using a `CUDA` device as the compute device:

[cuda_readme.rs]

[cuda_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cuda_readme.rs
```rust
use custos::{Buffer, CUDA};

fn main() -> custos::Result<()> {
    let device = CUDA::new(0)?;

    let mut a = Buffer::from((&device, [5, 3, 2, 4, 6, 2]));
    a.clear();

    assert_eq!(a.read(), [0; 6]);
    Ok(())
}
```

A lot more examples can be found in the 'tests' and 'examples' folder.
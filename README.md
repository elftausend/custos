![custos logo](assets/custos.png)

<hr/>

[![Crates.io version](https://img.shields.io/crates/v/custos.svg)](https://crates.io/crates/custos)
[![Docs](https://docs.rs/custos/badge.svg?version=0.4.6)](https://docs.rs/custos/0.4.6/custos/)
[![Rust](https://github.com/elftausend/custos/actions/workflows/rust.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/rust.yml)
[![GPU](https://github.com/elftausend/custos/actions/workflows/gpu.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/gpu.yml)
[![rust-clippy](https://github.com/elftausend/custos/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/elftausend/custos/actions/workflows/rust-clippy.yml)

A minimal OpenCL, CUDA and host CPU array manipulation engine / framework written in Rust.
This crate provides the tools for executing custom array operations with the CPU, as well as with CUDA and OpenCL devices.<br>
This guide demonstrates how operations can be implemented for the compute devices: [implement_operations.md](implement_operations.md)<br>
or to see it at a larger scale, look here: [custos-math]

[custos-math]: https://github.com/elftausend/custos-math

## Installation

Add "custos" as a dependency:
```toml
[dependencies]
custos = "0.4.6"

# to disable the default features (cuda, opencl) and use an own set of features:
#custos = {version = "0.4.6", default-features=false, features=["opencl"]}
```

Available features: 
- "opencl" ... adds OpenCL features, where the CLDevice (feature) is the most important one.
- "cuda" ... adds CUDA features. (CudaDevice)
- "realloc" ... disables caching for all devices.
- using no features at all ... CPU with BLAS

## [Examples]

These examples show how to use the implemented operations. <br>
custos only implements three buffer operations. These would be the write, read, and clear operations.<br>
On the other hand, [custos-math] implements a lot more operations, including Matrix operations for a custom Matrix struct.<br>
If you want to implement your own operations for all compute devices, look here: [implement_operations.md](implement_operations.md)

[examples]: https://github.com/elftausend/custos/tree/main/examples

Using the host CPU as the compute device:

[cpu_readme.rs]

[cpu_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cpu_readme.rs
```rust
use custos::{Buffer, ClearBuf, VecRead, CPU};

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

Using an OpenCL device as the compute device:

[cl_readme.rs]

[cl_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cl_readme.rs
```rust
use custos::{Buffer, CLDevice};

fn main() -> custos::Result<()> {
    let device = CLDevice::new(0)?;

    let mut a = Buffer::from((&device, [5, 3, 2, 4, 6, 2]));
    a.clear();

    assert_eq!(a.read(), [0; 6]);
    Ok(())
}

```

Using a CUDA device as the compute device:

[cuda_readme.rs]

[cuda_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cuda_readme.rs
```rust
use custos::{Buffer, CudaDevice};

fn main() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;

    let mut a = Buffer::from((&device, [5, 3, 2, 4, 6, 2]));
    a.clear();

    assert_eq!(a.read(), [0; 6]);
    Ok(())
}
```

A lot more examples can be found in the 'tests' and 'examples' folder.
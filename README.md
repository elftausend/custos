# custos

[![Crates.io version](https://img.shields.io/crates/v/custos.svg)](https://crates.io/crates/custos)
[![Docs](https://docs.rs/custos/badge.svg?version=0.1.2)](https://docs.rs/custos/0.1.2/custos/)

A minimal OpenCL, CUDA and host CPU array manipulation engine / framework.
It provides the tools needed to execute array operations with the CPU, as well as with CUDA and OpenCL devices.
This library demonstrates how the operations can be implemented for the compute devices: [custos-math]

[custos-math]: https://github.com/elftausend/custos-math

## Installation

Add "custos" as a dependency:
```toml
[dependencies]
custos = "0.1.2"

# to disable the default features (cuda, opencl):
#custos = {version = "0.1.2", default-features=false, features=["opencl", "safe"]}
```

Available features: 
- "opencl" ... adds OpenCL features, where the CLDevice (feature) is the most important one.
- "cuda" ... adds CUDA features. (CudaDevice)
- "safe" ... non-copy matrix and buffer. (safer)

## [Examples]

[examples]: https://github.com/elftausend/custos/tree/main/examples

Using the host CPU as the compute device:

[cpu_readme.rs]

[cpu_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cpu_readme.rs
```rust
use custos::{CPU, AsDev, VecRead, Buffer, ClearBuf};

fn main() {
    let device = CPU::new();
    let mut a = Buffer::from(( &device, [1, 2, 3, 4, 5, 6]));
    
    // specify device for operation
    device.clear(&mut a);
    assert_eq!(device.read(&a), [0; 6]);

    // select() ... sets CPU as 'global device' 
    // -> when device is not specified in an operation, the 'global device' is used
    let device = CPU::new().select();

    let mut a = Buffer::from(( &device, [1, 2, 3, 4, 5, 6]));

    // no need to specify the device
    a.clear();
    assert_eq!(a.read(), vec![0; 6]);
}
```

Using an OpenCL device as the compute device:

[cl_readme.rs]

[cl_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cl_readme.rs
```rust
use custos::{AsDev, Buffer, CLDevice};

fn main() -> custos::Result<()> {
    let device = CLDevice::new(0)?.select();
    
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
use custos::{CudaDevice, AsDev, Buffer};

fn main() -> custos::Result<()> {
    let device = CudaDevice::new(0)?.select();
    
    let mut a = Buffer::from((&device, [5, 3, 2, 4, 6, 2]));
    a.clear();

    assert_eq!(a.read(), [0; 6]);
    
    Ok(())
}
```
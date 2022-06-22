# custos

A minimal OpenCL, CUDA and host CPU array manipulation engine / framework.
It provides some matrix / buffer operations: matrix multiplication (BLAS, cuBLAS), element-wise arithmetic (vector addition, ...), set all elements to zero (or default value).
To use more operations: [custos-math]

[custos-math]: https://github.com/elftausend/custos-math

## Installation

Add "custos" as a dependency:
```toml
[dependencies]
custos = {git = "https://github.com/elftausend/custos.git", features=["opencl"]}
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
use custos::{CPU, AsDev, Matrix, BaseOps, VecRead};

fn main() {
    let device = CPU::new();
    let a = Matrix::from(( &device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let b = Matrix::from(( &device, (2, 3), [6, 5, 4, 3, 2, 1]));
    
    // specify device for operation
    let c = device.add(&a, &b);
    assert_eq!(device.read(&c), [7, 7, 7, 7, 7, 7]);

    // select() ... sets CPU as 'global device' 
    // -> when device is not specified in an operation, the 'global device' is used
    let device = CPU::new().select();

    let a = Matrix::from(( &device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let b = Matrix::from(( &device, (2, 3), [6, 5, 4, 3, 2, 1]));

    let c = a + b;
    assert_eq!(c.read(), vec![7, 7, 7, 7, 7, 7]);
}
```

Using an OpenCL device as the compute device:

[cl_readme.rs]

[cl_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cl_readme.rs
```rust
use custos::{CLDevice, Matrix, AsDev};

fn main() -> custos::Result<()> {
    let device = CLDevice::new(0)?.select();
    let a = Matrix::from((&device, 2, 3, [5, 3, 2, 4, 6, 2]));
    let b = Matrix::from((&device, 1, 6, [1, 4, 0, 2, 1, 3]));

    let c = a + b;
    assert_eq!(c.read(), [6, 7, 2, 6, 7, 5]);

    Ok(())
}
```

Using a CUDA device as the compute device:

[cuda_readme.rs]

[cuda_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cuda_readme.rs
```rust
use custos::{CudaDevice, Matrix, AsDev};

fn main() -> custos::Result<()> {
    let device = CudaDevice::new(0)?.select();
    let a = Matrix::from((&device, 2, 3, [5, 3, 2, 4, 6, 2]));
    let b = Matrix::from((&device, 1, 6, [1, 4, 0, 2, 1, 3]));

    let c = a + b;
    assert_eq!(c.read(), [6, 7, 2, 6, 7, 5]);

    Ok(())
}
```
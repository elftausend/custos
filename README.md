# custos

An experimental library for matrix operations that are calculated on the CPU or OpenCL devices.

## Example

```rust
use custos::{libs::{cpu::CPU, opencl::{CLDevice, api::OCLError}}, AsDev, Matrix, BaseOps, VecRead};

fn main() -> Result<(), OCLError> {
    //select() ... sets CPU as 'global device' 
    // -> when device is not specified in an operation, the 'global device' is used
    CPU.select();

    let a = Matrix::from( ((2, 3), &[1, 2, 3, 4, 5, 6] ));
    let b = Matrix::from( ((2, 3), &[6, 5, 4, 3, 2, 1] ));

    let c = a + b;
    assert_eq!(c.read(), vec![7, 7, 7, 7, 7, 7]);

    //device is specified in every operation
    let a = Matrix::from( (CPU, (2, 2), &[0.25, 0.5, 0.75, 1.] ) );
    let b = Matrix::from( (CPU, (2, 2), &[1., 2., 3., 4.,] ) );

    let c_cpu = CPU.mul(a, b);
    assert_eq!(CPU.read(c_cpu.data()), vec![0.25, 1., 2.25,  4.,]);

    //OpenCL device (GPU)
    CLDevice::get(0)?.select();

    let a = Matrix::from( ((2, 2), &[0.25f32, 0.5, 0.75, 1.] ) );
    let b = Matrix::from( ((2, 2), &[1., 2., 3., 4.,] ) );

    let c_cl = a * b;
    assert_eq!(CPU.read(c_cpu.data()), c_cl.read());

    Ok(())
}
```

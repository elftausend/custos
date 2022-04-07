
use custos::{libs::{cpu::CPU, opencl::CLDevice}, AsDev, Matrix, BaseOps, VecRead, Error};

fn main() -> Result<(), Error> {
    //select() ... sets CPU as 'global device' 
    // -> when device is not specified in an operation, the 'global device' is used
    let cpu = CPU::new().select();

    let a = Matrix::from(( &cpu, (2, 3), [1, 2, 3, 4, 5, 6]));
    let b = Matrix::from(( &cpu, (2, 3), [6, 5, 4, 3, 2, 1]));

    let c = a + b;
    assert_eq!(c.read(), vec![7, 7, 7, 7, 7, 7]);

    //device is specified in every operation
    let a = Matrix::from( ( &cpu, (2, 2), [0.25, 0.5, 0.75, 1.] ));
    let b = Matrix::from( ( &cpu, (2, 2), [1., 2., 3., 4.,] ));

    let c_cpu = cpu.mul(a, b);
    assert_eq!(cpu.read(c_cpu.data()), vec![0.25, 1., 2.25,  4.,]);

    //OpenCL device (GPU)
    let cl = CLDevice::get(0)?.select();

    let a = Matrix::from(( &cl, (2, 2), [0.25f32, 0.5, 0.75, 1.] ));
    let b = Matrix::from(( &cl, (2, 2), [1., 2., 3., 4.,] ));

    let c_cl = a * b;
    assert_eq!(cpu.read(c_cpu.data()), c_cl.read());

    Ok(())
}

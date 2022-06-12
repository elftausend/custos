use custos::{libs::{cpu::CPU, opencl::CLDevice}, AsDev, Matrix, BaseOps, VecRead, InternCPU};

fn main() -> custos::Result<()> {
    //select() ... sets CPU as 'global device' 
    // -> when device is not specified in an operation, the 'global device' is used
    let cpu = CPU::new().select();

    with_select(&cpu);
    specify_device(&cpu);

    using_opencl()
}

fn with_select(cpu: &InternCPU) {
    let a: Matrix<i32> = Matrix::from(( cpu, (2, 3), [1, 2, 3, 4, 5, 6]));
    let b = Matrix::from(( cpu, (2, 3), [6, 5, 4, 3, 2, 1]));

    let c = a + b;
    assert_eq!(c.read(), vec![7, 7, 7, 7, 7, 7]);
}

fn specify_device(cpu: &InternCPU) {
    //device is specified in every operation
    let a = Matrix::from( ( cpu, (2, 2), [0.25f32, 0.5, 0.75, 1.] ));
    let b = Matrix::from( ( cpu, (2, 2), [1., 2., 3., 4.,] ));

    let c_cpu = cpu.mul(&a, &b);
    assert_eq!(cpu.read(&c_cpu), vec![0.25, 1., 2.25,  4.,]);
}

fn using_opencl() -> custos::Result<()> {
    //OpenCL device (GPU)
    let cl = CLDevice::new(0)?.select();

    let a = Matrix::from(( &cl, (2, 2), [0.25f32, 0.5, 0.75, 1.] ));
    let b = Matrix::from(( &cl, (2, 2), [1., 2., 3., 4.] ));

    let c = a * b;
    assert_eq!(c.read(), vec![0.25, 1., 2.25,  4.,]);
    Ok(())
}
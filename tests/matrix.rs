use custos::{AsDev, libs::{cpu::CPU, opencl::{api::OCLError, CLDevice}}, Matrix, BaseOps, VecRead};

#[test]
fn test_matrix_read() {
    CPU.select();

    let matrix = Matrix::from(((2, 3), &[1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = matrix.read();
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
}

//safety ideas
#[test]
fn test_simple() -> Result<(), OCLError> {
    //device: Rc<>
    //when dropped: deallocate ?
    CPU.select();

    let a = Matrix::from( ((2, 3), &[1, 2, 3, 4, 5, 6] ));

    // "drop(device)" : a is still allocated

    let b = Matrix::from( ((2, 3), &[6, 5, 4, 3, 2, 1] ));

    let c = a + b;
    assert_eq!(c.read(), vec![7, 7, 7, 7, 7, 7]);

    let a = Matrix::from( (CPU, (2, 2), &[0.25, 0.5, 0.75, 1.] ) );
    let b = Matrix::from( (CPU, (2, 2), &[1., 2., 3., 4.,] ) );

    let c_cpu = CPU.mul(a, b);
    assert_eq!(CPU.read(c_cpu.data()), vec![0.25, 1., 2.25,  4.,]);

    CLDevice::get(0)?.select();

    let a = Matrix::from( ((2, 2), &[0.25f32, 0.5, 0.75, 1.] ) );
    let b = Matrix::from( ((2, 2), &[1., 2., 3., 4.,] ) );

    let c_cl = a * b;
    assert_eq!(CPU.read(c_cpu.data()), c_cl.read());

    Ok(())
}


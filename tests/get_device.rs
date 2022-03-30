
use custos::{AsDev, BaseDevice, get_device, libs::{cpu::CPU, opencl::CLDevice}, Matrix, VecRead, Error, BaseOps};

#[test]
fn test_matrix_read() -> Result<(), Error> {
    let device = CPU::new().select();

    let read = get_device!(VecRead, f32)?;

    let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.data());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);

    let device = CLDevice::get(0)?.select();

    let read = get_device!(VecRead, f32)?;

    let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.data());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
    
    let base_device = get_device!(BaseDevice, f32)?;
    assert_eq!(&read, &base_device.read(matrix.data()));
    Ok(())
}

#[test]
fn test_no_device() {
    let device = get_device!(BaseOps, f32);
    match device {
        Ok(_) => panic!("Should not panic, as no device is selected"),
        Err(_) => {},
    };

    {
        let device = CPU::new().select();

        let a = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
        let b = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    
        let c = device.add(a, b);
        assert_eq!(c.read(), vec![3.02, 12.246, 14., 10.42, 17.24, 9.53]);

        let device = get_device!(BaseOps, f32);
        match device {
            Ok(_) => {},
            Err(_) => panic!("Should not panic, as a device is selected"),
        };
    }

    match device {
        Ok(_) => panic!("Should not panic, as no device is selected"),
        Err(_) => {},
    };
}

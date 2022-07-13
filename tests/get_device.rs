use custos::{get_device, AsDev, Buffer, VecRead, CPU};

#[test]
fn get_device_test_cpu() {
    let device = CPU::new().select();

    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    let read_device = get_device!(VecRead<f32>).unwrap();
    assert_eq!(vec![1., 1.5, 0.14], read_device.read(&buf));
}

#[cfg(feature = "opencl")]
#[test]
fn get_device_test_cl() -> custos::Result<()> {
    use custos::CLDevice;

    let device = CLDevice::new(0)?.select();

    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    let read_device = get_device!(VecRead<f32>).unwrap();
    assert_eq!(vec![1., 1.5, 0.14], read_device.read(&buf));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn get_device_test_cu() -> custos::Result<()> {
    use custos::CudaDevice;

    let device = CudaDevice::new(0)?.select();

    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    let read_device = get_device!(VecRead<f32>).unwrap();
    assert_eq!(vec![1., 1.5, 0.14], read_device.read(&buf));
    Ok(())
}

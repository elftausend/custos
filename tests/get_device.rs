use custos::{get_device, Buffer, VecRead, CPU, AsDev};

#[test]
fn get_device_test_cpu() {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    let read_device = get_device!(device.as_dev(), VecRead<f32>);
    assert_eq!(vec![1., 1.5, 0.14], read_device.read(&buf));
}

#[cfg(feature = "opencl")]
#[test]
fn get_device_test_cl() -> custos::Result<()> {
    use custos::CLDevice;

    let device = CLDevice::new(0)?;

    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    let read_device = get_device!(VecRead<f32>).unwrap();
    assert_eq!(vec![1., 1.5, 0.14], read_device.read(&buf));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn get_device_test_cu() -> custos::Result<()> {
    use custos::CudaDevice;

    let device = CudaDevice::new(0)?;

    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    let read_device = get_device!(VecRead<f32>).unwrap();
    assert_eq!(vec![1., 1.5, 0.14], read_device.read(&buf));
    Ok(())
}

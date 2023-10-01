use custos::prelude::*;

#[cfg(feature = "cpu")]
#[test]
fn get_device_test_cpu() {
    let device = CPU::<Base>::new();

    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    let read = buf.device().read(&buf);
    assert_eq!(vec![1., 1.5, 0.14], read);
}

#[cfg(feature = "opencl")]
#[test]
fn get_device_test_cl() -> custos::Result<()> {
    use custos::OpenCL;

    let device = OpenCL::<Base>::new(chosen_cl_idx())?;
    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    assert_eq!(vec![1., 1.5, 0.14], buf.device().read(&buf));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn get_device_test_cu() -> custos::Result<()> {
    use custos::CUDA;

    let device = CUDA::<Base>::new(0)?;

    let buf = Buffer::from((&device, [1., 1.5, 0.14]));

    assert_eq!(vec![1., 1.5, 0.14], buf.device().read(&buf));
    Ok(())
}

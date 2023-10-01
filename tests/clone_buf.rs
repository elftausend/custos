use custos::prelude::*;

#[cfg(feature = "cpu")]
#[test]
fn test_buf_clone() {
    use custos::CloneBuf;

    let device = CPU::<Base>::new();
    let buf = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let cloned = device.clone_buf(&buf);
    assert_eq!(buf.as_slice(), cloned.as_slice());
}

#[cfg(feature = "cpu")]
#[test]
fn test_self_buf_clone() {
    let device = CPU::<Base>::new();
    let buf = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let cloned = buf.clone();
    assert_eq!(buf.as_slice(), cloned.as_slice());
}

#[cfg(feature = "opencl")]
#[test]
fn test_buf_clone_cl() -> custos::Result<()> {
    let device = custos::OpenCL::<Base>::new(chosen_cl_idx())?;

    let buf = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let cloned = device.clone_buf(&buf);
    assert_eq!(buf.read(), cloned.read());

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_buf_clone_cu() -> custos::Result<()> {
    let device = custos::CUDA::<Base>::new(0)?;

    let buf = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let cloned = device.clone_buf(&buf);
    assert_eq!(buf.read(), cloned.read());

    Ok(())
}

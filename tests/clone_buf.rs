use custos::{Buffer, CloneBuf, CPU};

#[test]
fn test_buf_clone() {
    let device = CPU::new();
    let buf = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let cloned = device.clone_buf(&buf);
    assert_eq!(buf.as_slice(), cloned.as_slice());
}

#[test]
fn test_self_buf_clone() {
    let device = CPU::new();
    let buf = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let cloned = buf.clone();
    assert_eq!(buf.as_slice(), cloned.as_slice());
}

#[cfg(feature = "opencl")]
#[test]
fn test_buf_clone_cl() -> custos::Result<()> {
    let device = custos::CLDevice::new(0)?;

    let buf = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let cloned = device.clone_buf(&buf);
    assert_eq!(buf.read(), cloned.read());

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_buf_clone_cu() -> custos::Result<()> {
    let device = custos::CudaDevice::new(0)?;

    let buf = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let cloned = device.clone_buf(&buf);
    assert_eq!(buf.read(), cloned.read());

    Ok(())
}

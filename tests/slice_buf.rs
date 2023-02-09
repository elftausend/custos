use custos::{Buffer, CopySlice, CPU};

#[test]
fn test_buf_slice() {
    let device = CPU::new();
    let source = Buffer::from((&device, [1., 2., 6., 2., 4.]));
    let actual = device.copy_slice(&source, 1..3);
    assert_eq!(actual.read(), &[2., 6.]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_buf_slice_cl() -> custos::Result<()> {
    let device = custos::OpenCL::new(0)?;
    let source = Buffer::from((&device, [1., 2., 6., 2., 4.]));
    let actual = device.copy_slice(&source, 1..3);
    assert_eq!(actual.read(), &[2., 6.]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_buf_clone_cu() -> custos::Result<()> {
    let device = custos::CUDA::new(0)?;
    let source = Buffer::from((&device, [1., 2., 6., 2., 4.]));
    let actual = device.copy_slice(&source, 1..3);
    assert_eq!(actual.read(), &[2., 6.]);
    Ok(())
}

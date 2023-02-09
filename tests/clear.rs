use custos::{Buffer, CPU};

#[cfg(feature = "cpu")]
#[test]
fn test_clear_cpu() {
    let device = CPU::new();

    let mut buf = Buffer::from((&device, [1., 2., 3., 4., 5., 6.]));
    assert_eq!(buf.read(), vec![1., 2., 3., 4., 5., 6.,]);
    buf.clear();
    assert_eq!(buf.read(), vec![0.; 6]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_clear_cl() -> Result<(), custos::Error> {
    use custos::OpenCL;

    let device = OpenCL::new(0)?;

    let mut buf = Buffer::from((&device, [1., 2., 3., 4., 5., 6.]));
    assert_eq!(buf.read(), vec![1., 2., 3., 4., 5., 6.,]);
    buf.clear();
    assert_eq!(buf.read(), vec![0.; 6]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_clear_cuda() -> Result<(), custos::Error> {
    use custos::CUDA;

    let device = CUDA::new(0)?;

    let mut buf = Buffer::from((&device, [1., 2., 3., 4., 5., 6.]));
    assert_eq!(buf.read(), vec![1., 2., 3., 4., 5., 6.,]);
    buf.clear();
    assert_eq!(buf.read(), vec![0.; 6]);
    Ok(())
}

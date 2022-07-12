use custos::{CPU, AsDev, Buffer};

#[test]
fn test_clear_cpu() {
    let device = CPU::new().select();
    
    let mut buf = Buffer::from((&device, [1., 2., 3., 4., 5., 6.,]));
    assert_eq!(buf.read(), vec![1., 2., 3., 4., 5., 6.,]);
    buf.clear();
    assert_eq!(buf.read(), vec![0.; 6]);
}

#[cfg(feature="opencl")]
#[test]
fn test_clear_cl() -> Result<(), custos::Error> {
    use custos::CLDevice;

    let device = CLDevice::new(0)?.select();
    
    let mut buf = Buffer::from((&device, [1., 2., 3., 4., 5., 6.,]));
    assert_eq!(buf.read(), vec![1., 2., 3., 4., 5., 6.,]);
    buf.clear();
    assert_eq!(buf.read(), vec![0.; 6]);
    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_clear_cuda() -> Result<(), custos::Error> {
    use custos::CudaDevice;

    let device = CudaDevice::new(0)?.select();
    
    let mut buf = Buffer::from((&device, [1., 2., 3., 4., 5., 6.,]));
    assert_eq!(buf.read(), vec![1., 2., 3., 4., 5., 6.,]);
    buf.clear();
    assert_eq!(buf.read(), vec![0.; 6]);
    Ok(())
}
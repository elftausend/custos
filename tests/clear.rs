use custos::{CPU, AsDev, Matrix};

#[test]
fn test_clear_cpu() {
    let device = CPU::new().select();
    
    let mut matrix = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
    assert_eq!(matrix.read(), vec![1., 2., 3., 4., 5., 6.,]);
    matrix.clear();
    assert_eq!(matrix.read(), vec![0.; 6]);
}

#[cfg(feature="opencl")]
#[test]
fn test_clear_cl() -> Result<(), custos::Error> {
    use custos::CLDevice;

    let device = CLDevice::new(0)?.select();
    
    let mut matrix = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
    assert_eq!(matrix.read(), vec![1., 2., 3., 4., 5., 6.,]);
    matrix.clear();
    assert_eq!(matrix.read(), vec![0.; 6]);
    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_clear_cuda() -> Result<(), custos::Error> {
    use custos::CudaDevice;

    let device = CudaDevice::new(0)?.select();
    
    let mut matrix = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
    assert_eq!(matrix.read(), vec![1., 2., 3., 4., 5., 6.,]);
    matrix.clear();
    assert_eq!(matrix.read(), vec![0.; 6]);
    Ok(())
}
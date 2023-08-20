use custos::prelude::*;

use custos_macro::stack_cpu_test;

//#[cfg(feature = "cpu")]
#[stack_cpu_test]
#[test]
fn test_clear_cpu() {
    let device = CPU::<custos::Base>::new();

    let mut buf = Buffer::with(&device, [1., 2., 3., 4., 5., 6.]);
    assert_eq!(buf.read(), [1., 2., 3., 4., 5., 6.,]);
    buf.clear();
    assert_eq!(buf.read(), [0.; 6]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_clear_cl() -> Result<(), custos::Error> {
    use custos::OpenCL;

    let device = OpenCL::<Base>::new(0)?;

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

    let device = CUDA::<Base>::new(0)?;

    let mut buf = Buffer::from((&device, [1., 2., 3., 4., 5., 6.]));
    assert_eq!(buf.read(), vec![1., 2., 3., 4., 5., 6.,]);
    buf.clear();
    assert_eq!(buf.read(), vec![0.; 6]);
    Ok(())
}

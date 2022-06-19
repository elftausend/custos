use custos::{CPU, Buffer, cpu::element_wise_op_mut};

fn slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    element_wise_op_mut(a, b, c, |a, b| a+b)
}

#[test]
fn test_deref_cpu() {
    let device = CPU::new();
    let a = Buffer::from((&device, [1., 2., 3., 4.,]));
    let b = Buffer::from((&device, [2., 3., 4., 5.,]));
    let mut c = Buffer::from((&device, [0.; 4]));

    slice_add(&a, &b, &mut c);

    assert_eq!(c.as_slice(), &[3., 5., 7., 9.,]);
}

#[cfg(feature="opencl")]
#[test]
#[should_panic]
fn test_deref_opencl() {
    use custos::CLDevice;

    let device = CLDevice::new(0).unwrap();
    if device.unified_mem() {
        panic!("the cpu ptr needs to be null")
    }
    let a = Buffer::from((&device, [1., 2., 3., 4.,]));
    let b = Buffer::from((&device, [2., 3., 4., 5.,]));
    let mut c = Buffer::from((&device, [0.; 4]));

    slice_add(&a, &b, &mut c);

    assert_eq!(c.as_slice(), &[3., 5., 7., 9.,]);
}

#[cfg(feature="cuda")]
#[test]
#[should_panic]
fn test_deref_cuda() {
    use custos::CudaDevice;

    let device = CudaDevice::new(0).unwrap();
    
    let a = Buffer::from((&device, [1., 2., 3., 4.,]));
    let b = Buffer::from((&device, [2., 3., 4., 5.,]));
    let mut c = Buffer::from((&device, [0.; 4]));

    slice_add(&a, &b, &mut c);

    assert_eq!(c.as_slice(), &[3., 5., 7., 9.,]);
}
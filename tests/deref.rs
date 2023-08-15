use custos::prelude::*;

fn slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    for i in 0..c.len() {
        c[i] = a[i] + b[i]
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_deref_cpu() {
    let device = CPU::<Base>::new();
    let a = Buffer::from((&device, [1., 2., 3., 4.]));
    let b = Buffer::from((&device, [2., 3., 4., 5.]));
    let mut c = Buffer::from((&device, [0.; 4]));

    slice_add(&a, &b, &mut c);

    assert_eq!(c.as_slice(), &[3., 5., 7., 9.,]);
}

/*
#[cfg(feature = "opencl")]
#[test]
#[should_panic]
fn test_deref_opencl() {
    use custos::OpenCL;

    let device = OpenCL::new(0).unwrap();
    if device.unified_mem() {
        panic!("the cpu ptr needs to be null")
    }
    let a = Buffer::from((&device, [1., 2., 3., 4.]));
    let b = Buffer::from((&device, [2., 3., 4., 5.]));
    let mut c = Buffer::from((&device, [0.; 4]));

    // does not compile!
    slice_add(&a, &b, &mut c);

    assert_eq!(c.as_slice(), &[3., 5., 7., 9.,]);
}
*/

/*
#[cfg(feature = "cuda")]
#[test]
// compile-time error instead
#[should_panic]
fn test_deref_cuda() {
    use custos::CUDA;

    let device = CUDA::new(0).unwrap();

    let a = Buffer::from((&device, [1., 2., 3., 4.]));
    let b = Buffer::from((&device, [2., 3., 4., 5.]));
    let mut c = Buffer::from((&device, [0.; 4]));

    slice_add(&a, &b, &mut c);

    assert_eq!(c.as_slice(), &[3., 5., 7., 9.,]);
}
*/

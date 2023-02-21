use custos::prelude::*;

#[cfg(feature = "cpu")]
#[test]
fn test_buf_slice_cpu() {
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

#[cfg(feature = "cpu")]
#[test]
fn test_buf_copy_slice_all_cpu() {
    let device = CPU::new();
    let source = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let mut dest = Buffer::new(&device, 10);

    device.copy_slice_all(&source, &mut dest, [(2..5, 7..10), (1..3, 3..5)]);

    assert_eq!(
        dest.read(),
        [0.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 6.0, 2.0, 4.0]
    );
}

#[cfg(feature = "opencl")]
#[test]
fn test_buf_copy_slice_all_opencl() {
    let device = OpenCL::new(0).unwrap();
    let source = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let mut dest = Buffer::new(&device, 10);

    device.copy_slice_all(&source, &mut dest, [(2..5, 7..10), (1..3, 3..5)]);

    assert_eq!(
        dest.read(),
        [0.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 6.0, 2.0, 4.0]
    );
}

#[cfg(feature = "cpu")]
#[should_panic]
#[test]
fn test_buf_copy_slice_all_cpu_out_of_bounds() {
    let device = CPU::new();
    let source = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let mut dest = Buffer::new(&device, 10);

    device.copy_slice_all(&source, &mut dest, [(2..5, 8..11), (1..3, 3..5)]);
}

#[cfg(feature = "opencl")]
#[should_panic]
#[test]
fn test_buf_copy_slice_all_opencl_out_of_bounds() {
    let device = OpenCL::new(0).unwrap();
    let source = Buffer::from((&device, [1., 2., 6., 2., 4.]));

    let mut dest = Buffer::new(&device, 10);

    device.copy_slice_all(&source, &mut dest, [(2..5, 8..11), (1..3, 3..5)]);
}

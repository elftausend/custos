#[cfg(feature = "cuda")]
#[test]
fn test_read_cuda() -> custos::Result<()> {
    use custos::{Buffer, Read, CUDA};

    let device = CUDA::new(0)?;
    let a = Buffer::from((&device, [3., 1., 3., 4.]));
    let read = device.read(&a);
    assert_eq!(vec![3., 1., 3., 4.,], read);
    Ok(())
}

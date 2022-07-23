#[cfg(feature = "cuda")]
#[test]
fn test_read_cuda() -> custos::Result<()> {
    use custos::{Buffer, CudaDevice, VecRead};

    let device = CudaDevice::new(0)?;
    let a = Buffer::from((&device, [3., 1., 3., 4.]));
    let read = device.read(&a);
    assert_eq!(vec![3., 1., 3., 4.,], read);
    Ok(())
}

#[test]
fn test_vec() {
    let _vec = Vec::from_iter(0..10);
}

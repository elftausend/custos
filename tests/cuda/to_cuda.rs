use custos::{Buffer, CudaDevice, CPU};

#[test]
fn test_to_cuda() -> custos::Result<()> {
    let cpu = CPU::new();

    let buf = Buffer::from((&cpu, [1., 2., 4., 1., 5., 1.]));

    let cuda = CudaDevice::new(0)?;
    let buf = buf.to_cuda(&cuda)?;

    assert_eq!(buf.read(), [1., 2., 4., 1., 5., 1.]);
    Ok(())
}

use custos::{Buffer, WriteBuf, CPU};

#[cfg(any(feature = "cuda", feature = "opencl"))]
use custos::VecRead;

#[test]
fn test_write_cpu() {
    let device = CPU::new();
    let mut buf = Buffer::new(&device, 5);
    device.write(&mut buf, &[1., 2., 3., 4., 5.]);
    assert_eq!(buf.as_slice(), &[1., 2., 3., 4., 5.])
}

#[cfg(feature = "opencl")]
#[test]
fn test_write_cl() -> custos::Result<()> {
    let device = custos::CLDevice::new(0)?;
    let mut buf = Buffer::new(&device, 5);
    device.write(&mut buf, &[1., 2., 3., 4., 5.]);
    assert_eq!(device.read(&buf), vec![1., 2., 3., 4., 5.]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_write_cuda() -> custos::Result<()> {
    let device = custos::CudaDevice::new(0)?;
    let mut buf = Buffer::new(&device, 5);
    device.write(&mut buf, &[1., 2., 3., 4., 5.]);
    assert_eq!(device.read(&buf), vec![1., 2., 3., 4., 5.]);
    Ok(())
}

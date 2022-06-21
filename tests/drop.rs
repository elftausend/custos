#[cfg(not(feature="safe"))]
use custos::{CPU, Device, Buffer};

#[cfg(not(feature="safe"))]
#[test]
fn test_drop_cpu() {
    let mut device = CPU::new();
    let buf = Buffer::from((&device, [4, 3, 1, 7, 8]));
    
    assert_eq!(device.inner.borrow().ptrs.len(), 1);

    device.drop(buf);
    assert_eq!(device.inner.borrow().ptrs.len(), 0);
}


#[cfg(not(feature="safe"))]
#[cfg(feature="opencl")]
#[test]
fn test_drop_cl() -> Result<(), custos::Error> {
    use custos::CLDevice;

    let mut device = CLDevice::new(0)?;
    let buf = Buffer::from((&device, [4, 3, 1, 7, 8]));
    
    assert_eq!(device.cl.borrow().ptrs.len(), 1);

    device.drop(buf);
    assert_eq!(device.cl.borrow().ptrs.len(), 0);
    Ok(())
}
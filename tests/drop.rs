use custos::{CPU, Device, Buffer};

#[test]
fn test_drop_cpu() {
    let mut device = CPU::new();
    let buf = Buffer::from((&device, [4, 3, 1, 7, 8]));
    
    assert_eq!(device.cpu.borrow().ptrs.len(), 1);

    device.drop(buf);
    assert_eq!(device.cpu.borrow().ptrs.len(), 0);
}


#[cfg(feature="opencl")]
#[test]
fn test_drop_cl() -> Result<(), custos::Error> {
    use custos::CLDevice;

    let mut device = CLDevice::get(0)?;
    let buf = Buffer::from((&device, [4, 3, 1, 7, 8]));
    
    assert_eq!(device.cl.borrow().ptrs.len(), 1);

    device.drop(buf);
    assert_eq!(device.cl.borrow().ptrs.len(), 0);
    Ok(())
}
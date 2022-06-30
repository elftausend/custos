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
    
    assert_eq!(device.inner.borrow().ptrs.len(), 1);

    device.drop(buf);
    assert_eq!(device.inner.borrow().ptrs.len(), 0);
    Ok(())
}

#[cfg(feature="safe")]
#[test]
fn test_drop_clone_safe() {
    use custos::{CPU, Buffer, VecRead};

    let device = CPU::new();
    let a = Buffer::from((&device, [4, 3, 1, 7, 8]));

    let b = a.clone();
    drop(a);
    assert_eq!(device.read(&b), vec![4, 3, 1, 7, 8]);
}

/*
#[test]
fn test_free_buffer() {
    let device = CPU::new();

    let a = Buffer::<f32>::from((&device, vec![3.12; 1000*10000]));
    assert_eq!(vec![3.12; a.len], a.as_slice());
    let vec: Vec<u16> = vec![1, 2, 5, 2, 9, 4];
    std::thread::sleep(std::time::Duration::from_secs_f32(2.5));
    
    drop(device);
    assert_eq!(vec![1, 2, 5, 2, 9, 4], vec);
    drop(vec);
    std::thread::sleep(std::time::Duration::from_secs_f32(5.4));
}
*/
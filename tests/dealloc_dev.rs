#[cfg(feature = "opencl")]
use custos::opencl::cl_device::CLDevice;
use custos::{libs::cpu::CPU, range, Buffer};

#[cfg(not(feature="realloc"))]
use custos::cpu::cpu_cached;

#[test]
fn test_rc_get_dev() {
    {
        let device = CPU::new();
        let mut a = Buffer::from((&device, [1., 2., 3., 4., 5., 6.]));

        for _ in range(100) {
            a.clear();
            assert_eq!(&[0.; 6], a.as_slice());
        }
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_dealloc_cl() -> custos::Result<()> {
    let device = CLDevice::new(0)?;

    let a = Buffer::from((&device, [1f32, 2., 3., 4., 5., 6.]));
    let b = Buffer::from((&device, [6., 5., 4., 3., 2., 1.]));

    drop(a);
    drop(b);
    drop(device);

    Ok(())
}

#[cfg(not(feature = "realloc"))]
#[test]
fn test_dealloc_device_cache_cpu() {
    let device = CPU::new();

    assert_eq!(device.cache.borrow().nodes.len(), 0);
    let a = cpu_cached::<f32>(&device, 10);
    assert_eq!(device.cache.borrow().nodes.len(), 1);

    drop(a);
    drop(device);
    //assert_eq!(device.cache.borrow().nodes.len(), 0);
}

#[cfg(not(feature = "realloc"))]
#[cfg(feature = "opencl")]
#[test]
fn test_dealloc_device_cache_cl() -> custos::Result<()> {
    use custos::opencl::cl_cached;

    let device = CLDevice::new(0)?;

    assert_eq!(device.cache.borrow().nodes.len(), 0);
    let a = cl_cached::<f32>(&device, 10);
    assert_eq!(device.cache.borrow().nodes.len(), 1);

    drop(a);
    drop(device);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_dealloc_device_cache_cu() -> custos::Result<()> {
    use custos::{cuda::cu_cached, CudaDevice};

    let device = CudaDevice::new(0)?;

    assert_eq!(device.cache.borrow().nodes.len(), 0);
    let a = cu_cached::<f32>(&device, 10);
    assert_eq!(device.cache.borrow().nodes.len(), 1);

    drop(a);
    drop(device);
    Ok(())
}

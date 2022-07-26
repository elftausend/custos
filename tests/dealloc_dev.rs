#[cfg(feature = "opencl")]
use custos::opencl::cl_device::CLDevice;
use custos::{cached, libs::cpu::CPU, range, AsDev, Buffer, cpu::CPU_CACHE};

#[test]
fn test_rc_get_dev() {
    {
        let device = CPU::new().select();
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
    let device = CLDevice::new(0)?.select();

    let _a = Buffer::from((&device, [1f32, 2., 3., 4., 5., 6.]));
    let _b = Buffer::from((&device, [6., 5., 4., 3., 2., 1.]));

    drop(device);

    Ok(())
}

// TODO: new test
#[test]
fn test_dealloc_device_cache_cpu() {
    let device = CPU::new().select();

    assert_eq!(CPU_CACHE.with(|cache| cache.borrow().nodes.len()), 0);
    let _a = cached::<f32>(10);
    assert_eq!(CPU_CACHE.with(|cache| cache.borrow().nodes.len()), 1);

    drop(device);
    assert_eq!(CPU_CACHE.with(|cache| cache.borrow().nodes.len()), 0);
}

// TODO: new test / access thread local in test
#[cfg(feature = "opencl")]
#[test]
fn test_dealloc_device_cache_cl() -> custos::Result<()> {
    use custos::opencl::CL_CACHE;

    let device = CLDevice::new(0)?.select();

    assert_eq!(CL_CACHE.with(|cache| cache.borrow().nodes.len()), 0);
    let _a = cached::<f32>(10);
    assert_eq!(CL_CACHE.with(|cache| cache.borrow().nodes.len()), 1);

    drop(device);
    assert_eq!(CL_CACHE.with(|cache| cache.borrow().nodes.len()), 0);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_dealloc_device_cache_cu() -> custos::Result<()> {
    use custos::{cuda::CudaCache, CudaDevice};

    let device = CudaDevice::new(0)?.select();

    let _a = cached::<f32>(10);
    assert_eq!(CudaCache::count(), 1);

    drop(device);
    assert_eq!(CudaCache::count(), 0);
    Ok(())
}

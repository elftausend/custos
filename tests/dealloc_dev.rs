#[cfg(feature = "opencl")]
use custos::opencl::cl_device::CLDevice;
use custos::{cached, cpu::CPUCache, libs::cpu::CPU, range, AsDev, Buffer};

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

#[test]
fn test_dealloc_device_cache_cpu() {
    let device = CPU::new().select();

    let _a = cached::<f32>(10);
    assert_eq!(CPUCache::count(), 1);

    drop(device);
    assert_eq!(CPUCache::count(), 0);
}

#[cfg(feature = "opencl")]
#[test]
fn test_dealloc_device_cache_cl() -> custos::Result<()> {
    use custos::opencl::CLCache;

    let device = CLDevice::new(0)?.select();

    let _a = cached::<f32>(10);
    assert_eq!(CLCache::count(), 1);

    drop(device);
    assert_eq!(CLCache::count(), 0);
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

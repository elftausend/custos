use custos::{CudaDevice, cuda::api::{cuDeviceGetAttribute, CUdevice_attribute}};


#[test]
fn test_cuda_ffi_unified_mem() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;

    let mut pi = 0;
    unsafe {
        cuDeviceGetAttribute(&mut pi, CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device.device().0);
    }

    println!("pi: {pi}");

    Ok(())
}
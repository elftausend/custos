use std::ptr::null_mut;

use super::{extern_c::{CudaPtr, cuMemAlloc_v2}, error::{CudaResult, CudaErrorKind}, cuInit, CUcontext, CUdevice, cuDeviceGet, cuCtxCreate_v2, cuMemFree_v2};

pub fn cinit(flags: u32) -> CudaResult<()> {
    unsafe { cuInit(flags).into() }
}

pub struct CudaIntDevice(CUdevice);

pub fn device(ordinal: i32) -> CudaResult<CudaIntDevice> {
    unsafe {
        let mut device = CudaIntDevice(0);
        cuDeviceGet(&mut device.0 as *mut i32, ordinal);
        Ok(device)
    }
}

pub struct Context(CUcontext);

pub fn create_context(device: CudaIntDevice) -> CudaResult<Context> {
    let mut context = Context(null_mut());
    unsafe {
        // TODO: Flags
        cuCtxCreate_v2(&mut context.0 as *mut CUcontext, 0, device.0).to_result()?;
    }
    Ok(context)
}

pub struct CudaMem(*mut *mut u64);

pub unsafe fn cmalloc<T>(len: usize) -> CudaResult<*mut CudaPtr> {
    let bytesize = len * core::mem::size_of::<T>();

    if bytesize == 0 {
        return Err(CudaErrorKind::InvalidAllocSize)
    }

    let mut ptr = null_mut();
    cuMemAlloc_v2(&mut ptr as *mut *mut u64 as *mut u64 , bytesize).to_result()?;
    Ok(ptr)
}

pub unsafe fn cfree(ptr: *mut CudaPtr) -> CudaResult<()> {
    cuMemFree_v2(ptr).to_result()
}
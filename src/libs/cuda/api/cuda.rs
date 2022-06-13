use std::{ptr::null_mut, ffi::c_void};

use crate::CUdeviceptr;

use super::{ffi::cuMemAlloc_v2, error::{CudaResult, CudaErrorKind}, cuInit, CUcontext, CUdevice, cuDeviceGet, cuCtxCreate_v2, cuMemFree_v2, cuDeviceGetCount, cuMemcpyHtoD_v2, cuMemcpyDtoH_v2};

pub fn cinit(flags: u32) -> CudaResult<()> {
    unsafe { cuInit(flags).into() }
}

#[derive(Debug)]
pub struct CudaIntDevice(CUdevice);

pub fn device_count() -> CudaResult<i32> {
    let mut count = 0;
    unsafe { cuDeviceGetCount(&mut count as *mut i32) }.to_result()?;
    Ok(count)
}

// TODO: cuda set device
pub fn device(ordinal: i32) -> CudaResult<CudaIntDevice> {
    if ordinal >= device_count()? {
        return Err(CudaErrorKind::InvalidDeviceIdx)
    } 

    let mut device = CudaIntDevice(0);
    unsafe { cuDeviceGet(&mut device.0 as *mut i32, ordinal) };
    Ok(device)
}

#[derive(Debug)]
pub struct Context(CUcontext);

pub fn create_context(device: &CudaIntDevice) -> CudaResult<Context> {
    let mut context = Context(null_mut());
    unsafe {
        // TODO: Flags
        cuCtxCreate_v2(&mut context.0 as *mut CUcontext, 0 , device.0).to_result()?;
    }
    Ok(context)
}


pub fn cumalloc<T>(len: usize) -> CudaResult<CUdeviceptr> {
    let bytes = len * core::mem::size_of::<T>();

    if bytes == 0 {
        return Err(CudaErrorKind::InvalidAllocSize)
    }

    let mut ptr: CUdeviceptr = 0;
    unsafe { cuMemAlloc_v2(&mut ptr , bytes).to_result()? };
    Ok(ptr)
}

pub fn cuwrite<T>(dst: CUdeviceptr, src_host: &[T]) -> CudaResult<()> {
    let bytes_to_copy = src_host.len() * std::mem::size_of::<T>();
    unsafe { cuMemcpyHtoD_v2(dst, src_host.as_ptr() as *const c_void, bytes_to_copy) }.to_result()?;
    Ok(())
}

pub fn curead<T>(dst_host: &mut [T], src: CUdeviceptr,) -> CudaResult<()> {
    let bytes_to_copy = dst_host.len() * std::mem::size_of::<T>();
    unsafe { cuMemcpyDtoH_v2(dst_host.as_mut_ptr() as *mut c_void, src, bytes_to_copy) }.to_result()?;
    Ok(())
}

pub unsafe fn cufree(ptr: CUdeviceptr) -> CudaResult<()> {
    cuMemFree_v2(ptr).to_result()
}
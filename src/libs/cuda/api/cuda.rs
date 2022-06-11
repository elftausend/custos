use std::ptr::null_mut;

use super::{extern_c::{CudaPtr, cuMemAlloc_v2}, error::{CudaResult, CudaErrorKind}, cuInit};

pub fn cinit(flags: u32) -> CudaResult<()> {
    unsafe { cuInit(flags).into() }
}

pub unsafe fn cmalloc<T>(len: usize) -> CudaResult<*mut CudaPtr> {
    let bytesize = len * core::mem::size_of::<T>();

    if bytesize == 0 {
        return Err(CudaErrorKind::InvalidAllocSize)
    }

    let ptr = null_mut();
    cuMemAlloc_v2(ptr, bytesize).to_result()?;
    Ok(ptr)
}

pub unsafe fn cfree(ptr: *mut CudaPtr) {
    
}
use crate::devices::cache::CacheType;
pub use blas::*;
pub use cpu_device::*;
use std::{
    alloc::Layout,
    mem::{align_of, size_of},
    ptr::null_mut,
};

mod blas;
mod cpu_device;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawCpuBuf {
    pub ptr: *mut u8,
    len: usize,
    align: usize,
    size: usize,
}

impl CacheType for RawCpuBuf {
    fn new<T>(ptr: (*mut T, *mut std::ffi::c_void, u64), len: usize) -> Self {
        RawCpuBuf {
            ptr: ptr.0 as *mut u8,
            len,
            align: align_of::<T>(),
            size: size_of::<T>(),
        }
    }

    fn destruct<T>(&self) -> (*mut T, *mut std::ffi::c_void, u64) {
        (self.ptr as *mut T, null_mut(), 0)
    }
}

impl Drop for RawCpuBuf {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(self.len*self.size, self.align).unwrap();
            std::alloc::dealloc(self.ptr, layout);
        }
    }
}

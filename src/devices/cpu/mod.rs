use crate::{/*devices::cache::CacheType,*/ Node, PtrType};
#[cfg(feature = "blas")]
pub use blas::*;
use core::{alloc::Layout, ptr::null_mut};
pub use cpu_device::*;

#[cfg(feature = "blas")]
mod blas;
mod cpu_device;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CPUPtr<T> {
    pub ptr: *mut T,
}

impl<T> Default for CPUPtr<T> {
    fn default() -> Self {
        Self { ptr: null_mut() }
    }
}

impl<T> PtrType<T> for CPUPtr<T> {
    #[inline]
    unsafe fn dealloc(&mut self, len: usize) {
        if self.ptr.is_null() {
            return;
        }
        let layout = Layout::array::<T>(len).unwrap();
        alloc::alloc::dealloc(self.ptr as *mut u8, layout);
    }

    #[inline]
    fn ptrs(&self) -> (*const T, *mut core::ffi::c_void, u64) {
        (self.ptr as *const T, null_mut(), 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut core::ffi::c_void, u64) {
        (self.ptr as *mut T, null_mut(), 0)
    }

    #[inline]
    unsafe fn from_ptrs(ptrs: (*mut T, *mut core::ffi::c_void, u64)) -> Self {
        CPUPtr { ptr: ptrs.0 }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawCpuBuf {
    pub ptr: *mut u8,
    len: usize,
    align: usize,
    size: usize,
    node: Node,
}

impl Drop for RawCpuBuf {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(self.len * self.size, self.align).unwrap();
            alloc::alloc::dealloc(self.ptr, layout);
        }
    }
}

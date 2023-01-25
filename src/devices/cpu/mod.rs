use crate::{CommonPtrs, PtrType, ShallowCopy};
#[cfg(feature = "blas")]
pub use blas::*;
use core::{alloc::Layout, mem::size_of, ptr::null_mut};
pub use cpu_device::*;
use std::alloc::handle_alloc_error;

use crate::flag::AllocFlag;

#[cfg(feature = "blas")]
mod blas;
mod cpu_device;
mod ops;

#[derive(PartialEq, Eq, Debug)]
pub struct CPUPtr<T> {
    pub ptr: *mut T,
    pub len: usize,
    pub flag: AllocFlag,
}

impl<T> CPUPtr<T> {
    pub fn new(len: usize, flag: AllocFlag) -> CPUPtr<T> {
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };

        // initialize block of memory
        for element in unsafe { std::slice::from_raw_parts_mut(ptr, len * size_of::<T>()) } {
            *element = 0;
        }

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        CPUPtr {
            ptr: ptr as *mut T,
            len,
            flag,
        }
    }
}

impl<T> Default for CPUPtr<T> {
    fn default() -> Self {
        Self {
            ptr: null_mut(),
            flag: AllocFlag::default(),
            len: 0,
        }
    }
}

impl<T> Drop for CPUPtr<T> {
    fn drop(&mut self) {
        if self.flag != AllocFlag::None {
            return;
        }

        if self.ptr.is_null() {
            return;
        }

        let layout = Layout::array::<T>(self.len).unwrap();

        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, layout);
        }
    }
}

impl<T> PtrType for CPUPtr<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.flag
    }
}

impl<T> CommonPtrs<T> for CPUPtr<T> {
    #[inline]
    fn ptrs(&self) -> (*const T, *mut core::ffi::c_void, u64) {
        (self.ptr as *const T, null_mut(), 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut core::ffi::c_void, u64) {
        (self.ptr as *mut T, null_mut(), 0)
    }
}

impl<T> ShallowCopy for CPUPtr<T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        CPUPtr {
            ptr: self.ptr,
            len: self.len,
            flag: AllocFlag::Wrapper,
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawCpuBuf {
    pub ptr: *mut u8,
    len: usize,
    align: usize,
    size: usize,
    flag: AllocFlag,
}

impl Drop for RawCpuBuf {
    fn drop(&mut self) {
        if self.flag != AllocFlag::Cache {
            return;
        }
        unsafe {
            let layout = Layout::from_size_align(self.len * self.size, self.align).unwrap();
            std::alloc::dealloc(self.ptr, layout);
        }
    }
}

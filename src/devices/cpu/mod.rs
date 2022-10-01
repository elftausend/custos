use crate::{devices::cache::CacheType, Node, PtrType, Alloc};
#[cfg(feature="blas")]
pub use blas::*;
pub use cpu_device::*;
use std::{
    alloc::Layout,
    mem::{align_of, size_of},
    ptr::null_mut,
};

#[cfg(feature="blas")]
mod blas;
mod cpu_device;

pub struct CPUPtr {
    pub ptr: *mut u8,
}

impl PtrType for CPUPtr {
    #[inline]
    unsafe fn alloc<T>(alloc: impl Alloc, len: usize) -> Self {
        CPUPtr {
            ptr: alloc.alloc::<T>(len).0 as *mut u8,
        }   
    }

    #[inline]
    unsafe fn dealloc<T>(&mut self, len: usize) {
        let layout = Layout::array::<T>(len).unwrap();
        std::alloc::dealloc(self.ptr, layout);
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

impl CacheType for RawCpuBuf {
    fn new<T>(ptr: (*mut T, *mut std::ffi::c_void, u64), len: usize, node: Node) -> Self {
        RawCpuBuf {
            ptr: ptr.0 as *mut u8,
            len,
            align: align_of::<T>(),
            size: size_of::<T>(),
            node,
        }
    }

    fn destruct<T>(&self) -> ((*mut T, *mut std::ffi::c_void, u64), Node) {
        ((self.ptr as *mut T, null_mut(), 0), self.node)
    }
}

impl Drop for RawCpuBuf {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(self.len * self.size, self.align).unwrap();
            std::alloc::dealloc(self.ptr, layout);
        }
    }
}

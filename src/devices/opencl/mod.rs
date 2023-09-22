//! The OpenCL module provides the OpenCL backend for custos.

use std::{ffi::c_void, ptr::null_mut};

pub use cl_device::{OpenCL, CL};
pub use kernel_cache::*;
pub use kernel_enqueue::*;

//pub mod api;
mod cl_device;
mod kernel_cache;
mod kernel_enqueue;

// #[cfg(unified_cl)]
mod unified;

mod ops;
pub use ops::*;

pub use min_cl::*;

use min_cl::api::release_mem_object;

// #[cfg(unified_cl)]
pub use unified::*;

//use self::api::release_mem_object;
use crate::{flag::AllocFlag, Buffer, CommonPtrs, HasId, Id, PtrType, ShallowCopy};

/// Another type for Buffer<'a, T, OpenCL, S>
pub type CLBuffer<'a, T, S = ()> = Buffer<'a, T, OpenCL, S>;

/// Reads the environment variable `CUSTOS_CL_DEVICE_IDX` and returns the value as a `usize`.
pub fn chosen_cl_idx() -> usize {
    std::env::var("CUSTOS_CL_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect(
            "Environment variable 'CUSTOS_CL_DEVICE_IDX' contains an invalid opencl device index!",
        )
}

/// The pointer used for `OpenCL` [`Buffer`](crate::Buffer)s
#[derive(Debug, PartialEq, Eq)]
pub struct CLPtr<T> {
    /// The pointer to the OpenCL memory object
    pub ptr: *mut c_void,
    /// Possibly a pointer to the host memory. Only active for devices with unified memory.
    pub host_ptr: *mut T,
    /// The number of elements allocated
    pub len: usize,
    /// The flag of the memory object
    pub flag: AllocFlag,
}

impl<T> Default for CLPtr<T> {
    #[inline]
    fn default() -> Self {
        Self {
            ptr: null_mut(),
            host_ptr: null_mut(),
            len: 0,
            flag: AllocFlag::default(),
        }
    }
}

impl<T> HasId for CLPtr<T> {
    #[inline]
    fn id(&self) -> Id {
        Id {
            id: self.ptr as u64,
            len: self.len,
        }
    }

    #[inline]
    unsafe fn set_id(&mut self, id: u64) {
        self.ptr = id as *mut u64 as *mut _
    }
}

impl<T> ShallowCopy for CLPtr<T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        CLPtr {
            ptr: self.ptr,
            host_ptr: self.host_ptr,
            len: self.len,
            flag: AllocFlag::Wrapper,
        }
    }
}

impl<T> PtrType for CLPtr<T> {
    #[inline]
    fn size(&self) -> usize {
        self.len
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.flag
    }
}

impl<T> Drop for CLPtr<T> {
    fn drop(&mut self) {
        if !matches!(self.flag, AllocFlag::None | AllocFlag::BorrowedCache) {
            return;
        }

        if self.ptr.is_null() {
            return;
        }
        unsafe {
            release_mem_object(self.ptr).unwrap();
        }
    }
}

impl<T> CommonPtrs<T> for CLPtr<T> {
    #[inline]
    fn ptrs(&self) -> (*const T, *mut c_void, u64) {
        (self.host_ptr, self.ptr, 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64) {
        (self.host_ptr, self.ptr, 0)
    }
}

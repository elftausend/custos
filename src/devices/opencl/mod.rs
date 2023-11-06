//! The OpenCL module provides the OpenCL backend for custos.

use core::ops::{Deref, DerefMut};
use std::{ffi::c_void, ptr::null_mut};

pub use cl_device::{OpenCL, CL};
pub use kernel_enqueue::*;

//pub mod api;
mod cl_device;
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
use crate::{flag::AllocFlag, CommonPtrs, HasId, HostPtr, Id, PtrType, ShallowCopy};

/// Another type for [`CLPtr`]
pub type CLBuffer<T> = CLPtr<T>;

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

#[cfg(unified_cl)]
impl<T> HostPtr<T> for CLPtr<T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.host_ptr
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.host_ptr
    }
}

impl<T> Deref for CLPtr<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for CLPtr<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
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

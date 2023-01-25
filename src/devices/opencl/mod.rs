use std::{ffi::c_void, ptr::null_mut};

pub use cl_device::{cl_cached, OpenCL, CL};
pub use kernel_cache::*;
pub use kernel_enqueue::*;

//pub mod api;
pub mod cl_device;
mod kernel_cache;
mod kernel_enqueue;

#[cfg(not(feature = "realloc"))]
#[cfg(unified_cl)]
mod unified;

mod ops;
pub use ops::*;

pub use min_cl::*;

use min_cl::api::release_mem_object;
#[cfg(unified_cl)]
#[cfg(not(feature = "realloc"))]
pub use unified::*;

//use self::api::release_mem_object;
use crate::{flag::AllocFlag, Buffer, CommonPtrs, PtrType, ShallowCopy};

pub type CLBuffer<'a, T, S = ()> = Buffer<'a, T, OpenCL, S>;

pub fn chosen_cl_idx() -> usize {
    std::env::var("CUSTOS_CL_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect(
            "Environment variable 'CUSTOS_CL_DEVICE_IDX' contains an invalid opencl device index!",
        )
}

#[derive(Debug, PartialEq, Eq)]
pub struct CLPtr<T> {
    pub ptr: *mut c_void,
    pub host_ptr: *mut T,
    pub len: usize,
    pub flag: AllocFlag,
}

impl<T> Default for CLPtr<T> {
    fn default() -> Self {
        Self {
            ptr: null_mut(),
            host_ptr: null_mut(),
            len: 0,
            flag: AllocFlag::default(),
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
    fn len(&self) -> usize {
        self.len
    }

    fn flag(&self) -> AllocFlag {
        self.flag
    }
}

impl<T> Drop for CLPtr<T> {
    fn drop(&mut self) {
        if self.flag != AllocFlag::None {
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

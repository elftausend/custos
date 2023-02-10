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

pub use min_cl::*;

use min_cl::api::release_mem_object;
#[cfg(unified_cl)]
#[cfg(not(feature = "realloc"))]
pub use unified::*;

//use self::api::release_mem_object;
use crate::{flag::AllocFlag, Buffer, CDatatype, CommonPtrs, PtrType, ShallowCopy};

pub type CLBuffer<'a, T> = Buffer<'a, T, OpenCL>;

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

    #[inline]
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

/// Sets the elements of an OpenCL Buffer to zero.
/// # Example
/// ```
/// use custos::{OpenCL, Buffer, Read, opencl::cl_clear};
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::new(0)?;
///     let mut lhs = Buffer::<i16, _>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(device.read(&lhs), vec![15, 30, 21, 5, 8]);
///
///     cl_clear(&device, &mut lhs);
///     assert_eq!(device.read(&lhs), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn cl_clear<T: CDatatype>(device: &OpenCL, lhs: &mut Buffer<T, OpenCL>) -> crate::Result<()> {
    let src = format!(
        "
        __kernel void clear(__global {datatype}* self) {{
            size_t id = get_global_id(0);
            self[id] = 0;
        }}
    ",
        datatype = T::as_c_type_str()
    );

    let gws = [lhs.len(), 0, 0];
    enqueue_kernel(device, &src, gws, None, &[lhs])?;
    Ok(())
}

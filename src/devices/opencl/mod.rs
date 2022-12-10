use std::{ffi::c_void, ptr::null_mut};

pub use cl_device::*;
pub use kernel_cache::*;
pub use kernel_enqueue::*;

pub mod api;
pub mod cl_device;
mod kernel_cache;
mod kernel_enqueue;

#[cfg(not(feature = "realloc"))]
#[cfg(unified_cl)]
mod unified;

#[cfg(unified_cl)]
#[cfg(not(feature = "realloc"))]
pub use unified::*;

use self::api::release_mem_object;
use crate::{Buffer, CDatatype, CommonPtrs, Dealloc, FromCommonPtrs};

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
}

impl<T> Default for CLPtr<T> {
    fn default() -> Self {
        Self {
            ptr: null_mut(),
            host_ptr: null_mut(),
        }
    }
}

impl<T> Dealloc<T> for CLPtr<T> {
    #[inline]
    unsafe fn dealloc(&mut self, _len: usize) {
        if self.ptr.is_null() {
            return;
        }
        release_mem_object(self.ptr).unwrap();
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

impl<T> FromCommonPtrs<T> for CLPtr<T> {
    #[inline]
    unsafe fn from_ptrs(ptrs: (*mut T, *mut c_void, u64)) -> Self {
        CLPtr {
            ptr: ptrs.1,
            host_ptr: ptrs.0,
        }
    }
}

/// Sets the elements of an OpenCL Buffer to zero.
/// # Example
/// ```
/// use custos::{OpenCL, Buffer, Read, opencl::cl_clear};
///
/// fn main() -> Result<(), custos::Error> {
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

    let gws = [lhs.len, 0, 0];
    enqueue_kernel(device, &src, gws, None, &[lhs])?;
    Ok(())
}

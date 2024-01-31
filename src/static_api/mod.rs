//! Exposes an API for static devices.
//! The usage is similiar to pytorch as `Buffer`s are moved to the gpu or another compute device via `.to_gpu`, `.to_cl`, ...
//! # Example
#![cfg_attr(any(feature = "cuda", feature = "opencl"), doc = "```")]
#![cfg_attr(not(any(feature = "cuda", feature = "opencl")), doc = "```ignore")]
//! use custos::buf;
//!
//! let buf = buf![2f32, 5., 1.].to_gpu();
//! assert_eq!(buf.read(), vec![2., 5., 1.]);
//!
//! ```

mod impl_buffer;
mod iter;
mod macros;
mod static_devices;
mod to_device;

pub use static_devices::*;

use crate::Device;

/// A trait that returns a device's respective static device.
pub trait StaticDevice: Device {
    /// Returns the static device. You can select the index of a static [`OpenCL`](crate::OpenCL) or [`CUDA`](crate::CUDA) device
    /// by setting the `CUSTOS_CL_DEVICE_IDX` or `CUSTOS_CU_DEVICE_IDX` environment variable.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, static_api::StaticDevice};
    ///
    /// let _static_cpu = CPU::as_static();
    ///
    /// ```
    fn as_static() -> &'static Self;
}

impl StaticDevice for crate::CPU {
    #[inline]
    fn as_static() -> &'static Self {
        static_cpu()
    }
}

/*#[cfg(feature = "stack")]
impl StaticDevice for crate::Stack {
    #[inline]
    fn as_static() -> &'static Self {
        &crate::Stack
    }
}*/

#[cfg(feature = "opencl")]
impl StaticDevice for crate::OpenCL {
    #[inline]
    fn as_static() -> &'static Self {
        static_opencl()
    }
}

#[cfg(feature = "cuda")]
impl StaticDevice for crate::CUDA {
    #[inline]
    fn as_static() -> &'static Self {
        static_cuda()
    }
}

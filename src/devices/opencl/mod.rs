//! The OpenCL module provides the OpenCL backend for custos.

pub use cl_device::{CL, OpenCL};
pub use kernel_enqueue::*;

//pub mod api;
mod cl_device;
mod kernel_enqueue;

// #[cfg(unified_cl)]
mod unified;

mod fusing;

mod ops;
pub use ops::*;

pub use min_cl::*;

mod cl_ptr;
pub use cl_ptr::CLPtr;

// #[cfg(unified_cl)]
pub use unified::*;

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

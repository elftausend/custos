//! The CPU module provides the CPU backend for custos.
#[cfg(feature = "blas")]
pub use blas::*;

pub use cpu_device::*;

#[cfg(feature = "blas")]
mod blas;
mod cpu_device;
mod cpu_ptr;
mod ops;

pub use cpu_ptr::*;
pub use ops::*;

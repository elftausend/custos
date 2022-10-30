#![cfg_attr(feature = "no-std", no_std)]

//! A minimal OpenCL, CUDA and host CPU array manipulation engine / framework written in Rust.
//! This crate provides the tools for executing custom array operations with the CPU, as well as with CUDA and OpenCL devices.<br>
//! This guide demonstrates how operations can be implemented for the compute devices: [implement_operations.md](implement_operations.md)<br>
//! or to see it at a larger scale, look here: [custos-math]
//!
//! ## [Examples]
//!
//! [examples]: https://github.com/elftausend/custos/tree/main/examples
//!
//! Using the host CPU as the compute device:
//!
//! [cpu_readme.rs]
//!
//! [cpu_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cpu_readme.rs
//!
//! ```rust
//! use custos::{CPU, ClearBuf, VecRead, Buffer};
//!
//! let device = CPU::new();
//! let mut a = Buffer::from(( &device, [1, 2, 3, 4, 5, 6]));
//!     
//! // specify device for operation
//! device.clear(&mut a);
//! assert_eq!(device.read(&a), [0; 6]);
//!
//! let device = CPU::new();
//!
//! let mut a = Buffer::from(( &device, [1, 2, 3, 4, 5, 6]));
//! a.clear();
//!
//! assert_eq!(a.read(), vec![0; 6]);
//! ```
extern crate alloc;
use alloc::vec::Vec;
use core::ffi::c_void;

//pub use libs::*;
pub use buffer::*;
pub use count::*;
pub use devices::*;
pub use error::*;
pub use graph::*;

pub use devices::cpu::CPU;
#[cfg(feature = "cuda")]
pub use devices::cuda::CUDA;
#[cfg(feature = "opencl")]
pub use devices::opencl::{CLDevice, OpenCL};

pub mod devices;

mod buffer;
mod count;
mod error;
mod graph;
mod op_traits;

#[cfg(feature = "static-api")]
pub mod static_api;

pub mod number;
pub use op_traits::*;

pub trait PtrType<T, const N: usize = 0> {
    /// # Safety
    /// The pointer must be a valid pointer.
    unsafe fn dealloc(&mut self, len: usize);

    fn ptrs(&self) -> (*const T, *mut c_void, u64);
    fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64);
    
    /// # Safety
    /// The pointer must be a valid pointer.
    unsafe fn from_ptrs(ptrs: (*mut T, *mut c_void, u64)) -> Self;
}

pub trait Device: Sized {
    type Ptr<U, const N: usize>: PtrType<U>;
    type Cache<const N: usize>: CacheAble<Self, N>;

    #[inline]
    fn retrieve<T, const N: usize>(&self, len: usize, add_node: impl AddGraph) -> Buffer<T, Self, N>
    where
        Self: Alloc<T, N>,
    {
        Self::Cache::retrieve(self, len, add_node)
    }
}

pub trait DevicelessAble<T, const N: usize = 0>: Alloc<T, N> {}

pub trait CPUCL: Device { 
    /// This is a device specific `as_slice()` function.
    /// As a 'StackArray' does not need to be checked for null.
    fn buf_as_slice<'a, T, const N: usize>(buf: &'a Buffer<T, Self, N>) -> &'a [T];
    fn buf_as_slice_mut<'a, T, const N: usize>(buf: &'a mut Buffer<T, Self, N>) -> &'a mut [T];
}

/// This trait is for allocating memory on the implemented device.
///
/// # Example
/// ```
/// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, GraphReturn, cpu::CPUPtr, PtrType};
///
/// let device = CPU::new();
/// let ptr = device.alloc(12);
///
/// let buf = Buffer {
///     ptr,
///     len: 12,
///     device: Some(&device),
///     flag: BufFlag::None,
///     node: device.graph().add_leaf(12),
/// };
/// assert_eq!(vec![0.; 12], device.read(&buf));
/// ```
pub trait Alloc<T, const N: usize = 0>: Device {
    /// Allocate memory on the implemented device.
    /// # Example
    /// ```
    /// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, GraphReturn, cpu::CPUPtr, PtrType};
    ///
    /// let device = CPU::new();
    /// let ptr = device.alloc(12);
    ///
    /// let buf = Buffer {
    ///     ptr,
    ///     len: 12,
    ///     device: Some(&device),
    ///     flag: BufFlag::None,
    ///     node: device.graph().add_leaf(12),
    /// };
    /// assert_eq!(vec![0.; 12], device.read(&buf));
    /// ```
    fn alloc(&self, len: usize) -> <Self as Device>::Ptr<T, N>;

    /// Allocate new memory with data
    /// # Example
    /// ```
    /// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, GraphReturn, cpu::CPUPtr, PtrType};
    ///
    /// let device = CPU::new();
    /// let ptr = device.with_slice(&[1, 5, 4, 3, 6, 9, 0, 4]);
    ///
    /// let buf = Buffer {
    ///     ptr,
    ///     len: 8,
    ///     device: Some(&device),
    ///     flag: BufFlag::None,
    ///     node: device.graph().add_leaf(8),
    /// };
    /// assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
    /// ```
    fn with_slice(&self, data: &[T]) -> <Self as Device>::Ptr<T, N>
    where
        T: Clone;

    /// If the vector `vec` was allocated previously, this function can be used in order to reduce the amount of allocations, which may be faster than using a slice of `vec`.
    /// #[inline]
    fn alloc_with_vec(&self, vec: Vec<T>) -> <Self as Device>::Ptr<T, N>
    where
        T: Clone,
    {
        self.with_slice(&vec)
    }

    #[inline]
    fn with_array(&self, array: [T; N]) -> <Self as Device>::Ptr<T, N>
    where
        T: Clone,
    {
        self.with_slice(&array)
    }
}

pub mod prelude {
    pub use crate::{
        cached, number::*, range, Buffer, CDatatype,
        CacheBuf, ClearBuf, Device, GraphReturn, VecRead, WriteBuf, CPU,
    };

    #[cfg(not(feature="no-std"))]
    pub use crate::{cache::CacheReturn, get_count, set_count, Cache};

    #[cfg(feature = "opencl")]
    pub use crate::opencl::OpenCL;

    #[cfg(feature = "opencl")]
    #[cfg(unified_cl)]
    #[cfg(not(feature = "realloc"))]
    pub use crate::opencl::{construct_buffer, to_unified};

    #[cfg(feature = "cuda")]
    pub use crate::cuda::CUDA;
}

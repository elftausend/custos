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
#![cfg_attr(feature = "cpu", doc = "```")]
#![cfg_attr(not(feature = "cpu"), doc = "```ignore")]
//! use custos::{CPU, ClearBuf, Read, Buffer};
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
use core::ffi::c_void;

//pub use libs::*;
pub use buffer::*;
pub use count::*;
pub use devices::*;

pub use error::*;

use flag::AllocFlag;
pub use graph::*;

#[cfg(feature = "cpu")]
pub use devices::cpu::CPU;

#[cfg(feature = "cuda")]
pub use devices::cuda::CUDA;
#[cfg(feature = "opencl")]
pub use devices::opencl::OpenCL;

#[cfg(feature = "wgpu")]
pub use devices::wgpu::WGPU;

#[cfg(feature = "stack")]
pub use devices::stack::Stack;

#[cfg(feature = "network")]
pub use devices::network::Network;

pub mod devices;

mod buffer;
mod count;
mod error;

pub mod flag;
mod graph;
mod op_traits;
mod shape;

#[cfg(feature = "static-api")]
pub mod static_api;

pub mod number;
pub use op_traits::*;
pub use shape::*;

pub trait PtrType {
    fn len(&self) -> usize;
    fn flag(&self) -> AllocFlag;
}

pub trait ShallowCopy {
    unsafe fn shallow(&self) -> Self;
}

pub trait CommonPtrs<T> {
    fn ptrs(&self) -> (*const T, *mut c_void, u64);
    fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64);
}

pub trait Device: Sized {
    type Ptr<U, S: Shape>: PtrType; //const B: usize, const C: usize
    type Cache: CacheAble<Self>;

    fn new() -> crate::Result<Self>;

    #[inline]
    fn retrieve<T, S: Shape>(&self, len: usize, add_node: impl AddGraph) -> Buffer<T, Self, S>
    where
        for<'a> Self: Alloc<'a, T, S>,
    {
        Self::Cache::retrieve(self, len, add_node)
    }
}

pub trait DevicelessAble<'a, T, S: Shape = ()>: Alloc<'a, T, S> {}

pub trait MainMemory: Device {
    fn as_ptr<T, S: Shape>(ptr: &Self::Ptr<T, S>) -> *const T;
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Ptr<T, S>) -> *mut T;
}

/// This trait is for allocating memory on the implemented device.
///
/// # Example
#[cfg_attr(feature = "cpu", doc = "```")]
#[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
/// use custos::{CPU, Alloc, Buffer, Read, flag::AllocFlag, GraphReturn, cpu::CPUPtr};
///
/// let device = CPU::new();
/// let ptr = Alloc::<f32>::alloc(&device, 12, AllocFlag::None);
///
/// let buf: Buffer = Buffer {
///     ptr,
///     device: Some(&device),
///     node: device.graph().add_leaf(12),
/// };
/// assert_eq!(vec![0.; 12], device.read(&buf));
/// ```
pub trait Alloc<'a, T, S: Shape = ()>: Device {
    /// Allocate memory on the implemented device.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Alloc, Buffer, Read, flag::AllocFlag, GraphReturn, cpu::CPUPtr};
    ///
    /// let device = CPU::new();
    /// let ptr = Alloc::<f32>::alloc(&device, 12, AllocFlag::None);
    ///
    /// let buf: Buffer = Buffer {
    ///     ptr,
    ///     device: Some(&device),
    ///     node: device.graph().add_leaf(12),
    /// };
    /// assert_eq!(vec![0.; 12], device.read(&buf));
    /// ```
    fn alloc(&'a self, len: usize, flag: AllocFlag) -> <Self as Device>::Ptr<T, S>;

    /// Allocate new memory with data
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Alloc, Buffer, Read, GraphReturn, cpu::CPUPtr};
    ///
    /// let device = CPU::new();
    /// let ptr = Alloc::<i32>::with_slice(&device, &[1, 5, 4, 3, 6, 9, 0, 4]);
    ///
    /// let buf: Buffer<i32, CPU> = Buffer {
    ///     ptr,
    ///     device: Some(&device),
    ///     node: device.graph().add_leaf(8),
    /// };
    /// assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
    /// ```
    fn with_slice(&'a self, data: &[T]) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone;

    /// If the vector `vec` was allocated previously, this function can be used in order to reduce the amount of allocations, which may be faster than using a slice of `vec`.
    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn alloc_with_vec(&'a self, vec: Vec<T>) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone,
    {
        self.with_slice(&vec)
    }

    #[inline]
    fn with_array(&'a self, array: S::ARR<T>) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone,
    {
        let stack_array = StackArray::<S, T>::from_array(array);
        self.with_slice(unsafe { stack_array.flatten() })
    }
}

#[cfg(not(unified_cl))]
pub const UNIFIED_CL_MEM: bool = false;

#[cfg(unified_cl)]
pub const UNIFIED_CL_MEM: bool = true;

#[cfg(feature = "macro")]
pub use custos_macro::impl_stack;

pub mod prelude {
    pub use crate::{
        cached, number::*, range, shape::*, Alloc, Buffer, CDatatype, CacheBuf, ClearBuf,
        CopySlice, Device, GraphReturn, MainMemory, Read, ShallowCopy, WithShape, WriteBuf,
    };

    #[cfg(feature = "cpu")]
    pub use crate::{cpu::cpu_cached, CPU};

    #[cfg(not(feature = "no-std"))]
    pub use crate::{cache::CacheReturn, get_count, set_count, Cache};

    #[cfg(feature = "opencl")]
    pub use crate::opencl::{enqueue_kernel, CLBuffer, OpenCL, CL};

    #[cfg(feature = "opencl")]
    #[cfg(unified_cl)]
    #[cfg(not(feature = "realloc"))]
    pub use crate::opencl::{cl_cached, construct_buffer, to_unified};

    #[cfg(feature = "stack")]
    pub use crate::stack::Stack;

    #[cfg(feature = "network")]
    pub use crate::network::{Network, NetworkArray};

    #[cfg(feature = "wgpu")]
    pub use crate::wgpu::{launch_shader, WGPU};

    #[cfg(feature = "cuda")]
    pub use crate::cuda::{launch_kernel1d, CUBuffer, CU, CUDA};
}

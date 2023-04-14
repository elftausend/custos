#![warn(missing_docs)]
#![cfg_attr(feature = "no-std", no_std)]

//! A minimal OpenCL, WGPU, CUDA and host CPU array manipulation engine / framework written in Rust.
//! This crate provides the tools for executing custom array operations with the CPU, as well as with CUDA, WGPU and OpenCL devices.<br>
//! This guide demonstrates how operations can be implemented for the compute devices: [implement_operations.md](implement_operations.md)<br>
//! or to see it at a larger scale, look here [custos-math] or here [sliced].
//!
//! [custos-math]: https://github.com/elftausend/custos-math
//! [sliced]: https://github.com/elftausend/sliced
//!
//! ## [Examples]
//!
//! custos only implements four `Buffer` operations. These would be the `write`, `read`, `copy_slice` and `clear` operations,
//! however, there are also [unary] (device only) operations.<br>
//! On the other hand, [custos-math] implements a lot more operations, including Matrix operations for a custom Matrix struct.<br>
//!
//! [examples]: https://github.com/elftausend/custos/tree/main/examples
//! [unary]: https://github.com/elftausend/custos/blob/main/src/unary.rs
//!
//! Implement an operation for `CPU`:
//! If you want to implement your own operations for all compute devices, consider looking here: [implement_operations.md](implement_operations.md)
//!
#![cfg_attr(feature = "cpu", doc = "```")]
#![cfg_attr(not(feature = "cpu"), doc = "```ignore")]
//! use std::ops::Mul;
//! use custos::prelude::*;
//!
//! pub trait MulBuf<T, S: Shape = (), D: Device = Self>: Sized + Device {
//!     fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
//! }
//!
//! impl<T, S, D> MulBuf<T, S, D> for CPU
//! where
//!     T: Mul<Output = T> + Copy,
//!     S: Shape,
//!     D: MainMemory,
//! {
//!     fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, CPU, S> {
//!         let mut out = self.retrieve(lhs.len(), (lhs, rhs));
//!
//!         for ((lhs, rhs), out) in lhs.iter().zip(&*rhs).zip(&mut out) {
//!             *out = *lhs * *rhs;
//!         }
//!
//!         out
//!     }
//! }
//! ```
//!
//! A lot more usage examples can be found in the [tests] and [examples] folder.
//!
//! [tests]: https://github.com/elftausend/custos/tree/main/tests
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

#[cfg(feature = "autograd")]
pub use autograd::*;

pub use unary::*;

#[cfg(feature = "cpu")]
#[macro_use]
pub mod exec_on_cpu;

pub mod devices;

mod buffer;
mod count;
mod error;

pub mod flag;
mod graph;
mod op_traits;
mod shape;
mod two_way_ops;
mod unary;

#[cfg(feature = "static-api")]
pub mod static_api;

#[cfg(feature = "autograd")]
pub mod autograd;
pub mod number;
pub use op_traits::*;
pub use shape::*;
pub use two_way_ops::*;

#[cfg(feature = "autograd")]
#[cfg(feature = "opt-cache")]
compile_error!("The `autograd` and `opt-cache` feature are currently incompatible. 
This is because the logic for detecting if a forward buffer is used during gradient calculation isn't implemented yet.");

#[cfg(feature = "autograd")]
#[cfg(feature = "realloc")]
compile_error!("The `autograd` and `realloc` feature are incompatible. 
The automatic differentiation system requires caching of buffers, which is deactivated when using the `realloc` feature.");

#[cfg(all(feature = "realloc", feature = "opt-cache"))]
compile_error!("A typical 'cache' does not exist when the `realloc` feature is enabled.");

/// This trait is implemented for every pointer type.
pub trait PtrType {
    /// Returns the element count.
    fn size(&self) -> usize;
    /// Returns the [`AllocFlag`].
    fn flag(&self) -> AllocFlag;
}

/// Used to shallow-copy a pointer. Use is discouraged.
pub trait ShallowCopy {
    /// # Safety
    /// Shallow copies of pointers may live longer than the corresponding resource.
    unsafe fn shallow(&self) -> Self;
}

/// custos v5 compatibility for "common pointers".
/// The commmon pointers contain the following pointers: host, opencl and cuda
pub trait CommonPtrs<T> {
    /// Returns the "immutable" common pointers.
    fn ptrs(&self) -> (*const T, *mut c_void, u64);
    /// Returns the mutable common pointers.
    fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64);
}

/// This trait is the base trait for every device.
pub trait Device: Sized + 'static {
    /// The type of the pointer that is used for `Buffer`.
    type Ptr<U, S: Shape>: PtrType;
    /// The type of the cache.
    type Cache: CacheAble<Self>;
    //type Tape: ;

    /// Creates a new device.
    fn new() -> crate::Result<Self>;

    /// Creates a new [`Buffer`] using `A`.
    ///
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Device};
    ///
    /// let device = CPU::new();
    /// let buf = device.buffer([5, 4, 3]);
    ///
    /// assert_eq!(buf.read(), [5, 4, 3]);
    /// ```
    fn buffer<'a, T, S: Shape, A>(&'a self, arr: A) -> Buffer<'a, T, Self, S>
    where
        Buffer<'a, T, Self, S>: From<(&'a Self, A)>,
    {
        Buffer::from((self, arr))
    }

    /// May allocate a new [`Buffer`] or return an existing one.
    /// It may use the cache count provided by the cache count (identified by [`Ident`]). <br>
    /// This depends on the type of cache and enabled features. <br>
    /// With the `realloc` feature enabled, it is guaranteed that the returned `Buffer` is newly allocated and freed every time.
    ///
    /// # Example
    #[cfg_attr(all(feature = "cpu", not(feature = "realloc")), doc = "```")]
    #[cfg_attr(all(not(feature = "cpu"), feature = "realloc"), doc = "```ignore")]
    /// use custos::{Device, CPU, set_count};
    ///
    /// let device = CPU::new();
    ///
    /// let buf = device.retrieve::<f32, ()>(10, ());
    ///
    /// // unsafe, because the next .retrieve call will then return the same buffer
    /// unsafe { set_count(0) }
    ///
    /// let buf_2 = device.retrieve::<f32, ()>(10, ());
    ///
    /// assert_eq!(buf.ptr.ptr, buf_2.ptr.ptr);
    ///
    /// ```
    #[inline]
    fn retrieve<T, S: Shape>(&self, len: usize, add_node: impl AddGraph) -> Buffer<T, Self, S>
    where
        for<'a> Self: Alloc<'a, T, S>,
    {
        Self::Cache::retrieve(self, len, add_node)
    }

    /// May return an existing buffer using the provided [`Ident`].
    /// This function panics if no buffer with the provided `Ident` exists.
    ///
    /// # Safety
    /// This function is unsafe because it is possible to return multiple [`Buffer`] with `Ident` that share the same memory.
    /// If this function is called twice with the same `Ident`, the returned `Buffer` will be the same.
    /// Even though the return `Buffer`s are owned, this does not lead to double-frees (see [`AllocFlag`]).
    #[cfg(feature = "autograd")]
    #[inline]
    unsafe fn get_existing_buf<T, S: Shape>(&self, ident: Ident) -> Buffer<T, Self, S> {
        Self::Cache::get_existing_buf(self, ident).expect("A matching Buffer does not exist.")
    }

    /// Removes a `Buffer` with the provided [`Ident`] from the cache.
    /// This function is internally called when a `Buffer` with [`AllocFlag`] `None` is dropped.
    #[cfg(not(feature = "no-std"))]
    #[inline]
    fn remove(&self, ident: Ident) {
        Self::Cache::remove(self, ident);
    }

    /// Adds a pointer that was allocated by [`Alloc`] to the cache and returns a new corresponding [`Ident`].
    /// This function is internally called when a `Buffer` with [`AllocFlag`] `None` is created.
    #[cfg(not(feature = "no-std"))]
    #[inline]
    fn add_to_cache<T, S: Shape>(&self, ptr: &Self::Ptr<T, S>) -> Option<Ident> {
        Self::Cache::add_to_cache(self, ptr)
    }
}

/// All type of devices that can create [`Buffer`]s
pub trait DevicelessAble<'a, T, S: Shape = ()>: Alloc<'a, T, S> {}

/// Devices that can access the main memory / RAM of the host.
pub trait MainMemory: Device {
    /// Returns the respective immutable host memory pointer
    fn as_ptr<T, S: Shape>(ptr: &Self::Ptr<T, S>) -> *const T;
    /// Returns the respective mutable host memory pointer
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
///     ident: None,
///     ptr,
///     device: Some(&device),
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
    ///     ident: None,
    ///     ptr,
    ///     device: Some(&device),
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
    ///     ident: None,
    ///     ptr,
    ///     device: Some(&device),
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

    /// Allocates a pointer with the array provided by the `S:`[`Shape`] generic.
    /// By default, the array is flattened and then passed to [`Alloc::with_slice`].
    #[inline]
    fn with_array(&'a self, array: S::ARR<T>) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone,
    {
        let stack_array = StackArray::<S, T>::from_array(array);
        self.with_slice(stack_array.flatten())
    }
}

/// If the `autograd` feature is enabled, then this will be implemented for all types that implement [`TapeReturn`].
/// On the other hand, if the `autograd` feature is disabled, no [`Tape`] will be returneable.
#[cfg(feature = "autograd")]
pub trait MayTapeReturn: crate::TapeReturn {}
#[cfg(feature = "autograd")]
impl<D: crate::TapeReturn> MayTapeReturn for D {}

/// If the `autograd` feature is enabled, then this will be implemented for all types that implement [`TapeReturn`].
/// On the other hand, if the `autograd` feature is disabled, no [`Tape`] will be returneable.
#[cfg(not(feature = "autograd"))]
pub trait MayTapeReturn {}
#[cfg(not(feature = "autograd"))]
impl<D> MayTapeReturn for D {}

/// If the OpenCL device selected by the environment variable `CUSTOS_CL_DEVICE_IDX` supports unified memory, then this will be `true`.
/// In your case, this is `false`.
#[cfg(not(unified_cl))]
pub const UNIFIED_CL_MEM: bool = false;

/// If the OpenCL device selected by the environment variable `CUSTOS_CL_DEVICE_IDX` supports unified memory, then this will be `true`.
/// In your case, this is `true`.
#[cfg(unified_cl)]
pub const UNIFIED_CL_MEM: bool = true;

#[cfg(feature = "macro")]
pub use custos_macro::impl_stack;


/// A dummy CPU. This only exists to make the code compile when the `cpu` feature is disabled
/// because the CPU is the default type `D` for [`Buffer`]s.
#[cfg(not(feature = "cpu"))]
pub struct CPU {
    _uncreateable: (),
}

#[cfg(not(feature = "cpu"))]
impl Device for CPU {
    type Ptr<U, S: Shape> = crate::Num<U>;

    type Cache = ();

    fn new() -> crate::Result<Self> {
        #[cfg(feature = "no-std")]
        {
            unimplemented!("CPU is not available. Enable the `cpu` feature to use the CPU.")
        }

        #[cfg(not(feature = "no-std"))]
        Err(crate::DeviceError::CPUDeviceNotAvailable.into())
    }
}


pub mod prelude {
    //! Typical imports for using custos.

    pub use crate::{
        number::*, range, shape::*, Alloc, Buffer, CDatatype, ClearBuf, CopySlice, Device,
        GraphReturn, Ident, MainMemory, MayTapeReturn, Read, ShallowCopy, WithShape, WriteBuf,
        MayToCLSource
    };

    #[cfg(feature = "cpu")]
    pub use crate::{exec_on_cpu::*, CPU};

    #[cfg(not(feature = "no-std"))]
    pub use crate::{cache::CacheReturn, get_count, set_count, Cache};

    #[cfg(feature = "opencl")]
    pub use crate::opencl::{enqueue_kernel, CLBuffer, OpenCL, CL};

    #[cfg(feature = "opencl")]
    #[cfg(unified_cl)]
    #[cfg(not(feature = "realloc"))]
    pub use crate::opencl::{construct_buffer, to_cached_unified};

    #[cfg(feature = "stack")]
    pub use crate::stack::Stack;

    #[cfg(feature = "network")]
    pub use crate::network::{Network, NetworkArray};

    #[cfg(feature = "wgpu")]
    pub use crate::wgpu::{launch_shader, WGPU};

    #[cfg(feature = "cuda")]
    pub use crate::cuda::{launch_kernel1d, CUBuffer, CU, CUDA};
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "cpu")]
    #[test]
    fn test_buffer_from_device() {
        use crate::{Device, CPU};

        let device = CPU::new();
        let buf = device.buffer([1, 2, 3]);

        assert_eq!(buf.read(), [1, 2, 3])
    }
}

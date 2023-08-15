// #![warn(missing_docs)]
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
pub use devices::*;

pub use error::*;

use flag::AllocFlag;

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

pub use unary::*;

#[cfg(feature = "cpu")]
#[macro_use]
pub mod exec_on_cpu;

pub mod devices;

mod buffer;
mod error;

mod cache;
mod device_traits;
mod features;
pub mod flag;
// mod graph;
mod hooks;
mod id;
mod modules;
mod op_traits;
mod parents;
mod ptr_conv;
mod shape;
mod two_way_ops;
mod unary;

pub use cache::*;
pub use device_traits::*;
pub use features::*;
pub use hooks::*;
pub use id::*;
pub use modules::*;
pub use parents::*;
pub use ptr_conv::*;

#[cfg(feature = "static-api")]
pub mod static_api;

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

#[cfg(test)]
pub fn location() -> &'static core::panic::Location<'static> {
    core::panic::Location::caller()
}

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

/// All type of devices that can create [`Buffer`]s
pub trait DevicelessAble<'a, T, S: Shape = ()>: Alloc<T> {}

/// Devices that can access the main memory / RAM of the host.
pub trait MainMemory: Device {
    /// Returns the respective immutable host memory pointer
    fn as_ptr<T, S: Shape>(ptr: &Self::Data<T, S>) -> *const T;
    /// Returns the respective mutable host memory pointer
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Data<T, S>) -> *mut T;
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
pub use custos_macro::*;

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
        device_traits::*, features::*, modules::*, number::*, shape::*, Alloc, Buffer, CDatatype,
        ClearBuf, CloneBuf, CopySlice, Device, MainMemory, MayTapeReturn, MayToCLSource, Read,
        ShallowCopy, WithShape, WriteBuf,
    };

    #[cfg(feature = "cpu")]
    pub use crate::{exec_on_cpu::*, CPU};

    // TODO
    // #[cfg(not(feature = "no-std"))]
    // pub use crate::{cache::CacheReturn, get_count, set_count, Cache};

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
    pub use crate::cuda::{launch_kernel1d, CUDA};
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "cpu")]
    #[test]
    fn test_buffer_from_device() {
        use crate::{Base, Device, CPU};

        let device = CPU::<Base>::new();
        let buf = device.buffer([1, 2, 3]);

        assert_eq!(buf.read(), [1, 2, 3])
    }
}

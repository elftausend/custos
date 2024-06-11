// #![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
// A compute kernel launch may wants to modify memory. Clippy does not know this.
// To declare that a value is mutated, a "needless" mutable reference is used.
#![allow(clippy::needless_pass_by_ref_mut)]

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
//! use custos::prelude::*;
//! use std::ops::{Deref, Mul};
//!
//! pub trait MulBuf<T: Unit, S: Shape = (), D: Device = Self>: Sized + Device {
//!     fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
//! }
//!
//! impl<Mods, T, S, D> MulBuf<T, S, D> for CPU<Mods>
//! where
//!     Mods: Retrieve<Self, T, S>,
//!     T: Unit + Mul<Output = T> + Copy + 'static,
//!     S: Shape,
//!     D: Device,
//!     D::Base<T, S>: Deref<Target = [T]>,
//! {
//!     fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
//!         let mut out = self.retrieve(lhs.len(), (lhs, rhs)).unwrap(); // unwrap or return error (update trait)
//!
//!         for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(&mut out) {
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

#[cfg(feature = "stack")]
pub use devices::stack::Stack;

#[cfg(feature = "nnapi")]
pub use devices::nnapi::{AsOperandCode, NnapiDevice};
#[cfg(feature = "vulkan")]
pub use devices::vulkan::Vulkan;

pub use unary::*;

#[cfg(feature = "std")]
pub use boxed_shallow_copy::*;

#[cfg(feature = "cpu")]
#[macro_use]
pub mod exec_on_cpu;

pub mod devices;

mod buffer;
mod error;

mod cache;

pub mod features;
pub mod flag;
// mod graph;
#[cfg(feature = "std")]
mod boxed_shallow_copy;
pub mod hooks;
mod id;
mod layer_management;
pub mod modules;
mod op_hint;
mod op_traits;
mod parents;
mod ptr_conv;
mod range;
mod shape;
mod two_way_ops;
mod unary;
mod update_args;
mod wrapper;

pub use cache::*;
pub use features::*;
pub use hooks::*;
pub use id::*;
pub use layer_management::*;
pub use modules::*;
pub use number::*;
pub use parents::*;
pub use ptr_conv::*;
pub use range::*;
pub use update_args::*;
pub use wrapper::*;

#[cfg(not(feature = "cpu"))]
pub mod dummy_cpu;

#[cfg(not(feature = "cpu"))]
pub use dummy_cpu::*;

#[cfg(feature = "static-api")]
pub mod static_api;

pub mod number;
pub use op_traits::*;
pub use shape::*;
pub use two_way_ops::*;

pub const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

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
    unsafe fn set_flag(&mut self, flag: AllocFlag);
}

pub trait HostPtr<T>: PtrType {
    fn ptr(&self) -> *const T;
    fn ptr_mut(&mut self) -> *mut T;

    #[inline]
    unsafe fn as_slice(&self) -> &[T] {
        core::slice::from_raw_parts(self.ptr(), self.size())
    }

    #[inline]
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        core::slice::from_raw_parts_mut(self.ptr_mut(), self.size())
    }
}

/// Minimum requirements for an element inside a Buffer.
pub trait Unit: Sync {}

impl<T: Sync> Unit for T {}

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
pub trait DevicelessAble<'a, T: Unit, S: Shape = ()>: Alloc<T> {}

/// If the `autograd` feature is enabled, then this will be implemented for all types that implement [`TapeActions`].
/// On the other hand, if the `autograd` feature is disabled, no [`Tape`] will be returneable.
#[cfg(feature = "autograd")]
pub trait MayTapeActions: TapeActions {}
#[cfg(feature = "autograd")]
impl<D: crate::TapeActions> MayTapeActions for D {}

/// If the `autograd` feature is enabled, then this will be implemented for all types that implement [`TapeReturn`].
/// On the other hand, if the `autograd` feature is disabled, no [`Tape`] will be returneable.
#[cfg(not(feature = "autograd"))]
pub trait MayTapeActions {}
#[cfg(not(feature = "autograd"))]
impl<D> MayTapeActions for D {}

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

#[cfg(feature = "std")]
pub(crate) type Buffers<B> =
    std::collections::HashMap<UniqueId, B, std::hash::BuildHasherDefault<NoHasher>>;

pub mod prelude {
    //! Typical imports for using custos.

    pub use crate::{
        devices::*, features::*, modules::*, number::*, shape::*, Alloc, Buffer, CDatatype,
        ClearBuf, CloneBuf, CopySlice, Device, Error, HasId, HostPtr, MayToCLSource, Read,
        ShallowCopy, Unit, WithShape, WriteBuf,
    };

    #[cfg(feature = "cpu")]
    pub use crate::{exec_on_cpu::*, CPU};

    #[cfg(feature = "opencl")]
    pub use crate::opencl::{chosen_cl_idx, enqueue_kernel, CLBuffer, OpenCL, CL};

    #[cfg(feature = "opencl")]
    #[cfg(unified_cl)]
    pub use crate::{opencl::construct_buffer, UnifiedMemChain};

    #[cfg(feature = "stack")]
    pub use crate::stack::Stack;

    #[cfg(feature = "nnapi")]
    pub use crate::nnapi::NnapiDevice;

    #[cfg(feature = "cuda")]
    pub use crate::cuda::{chosen_cu_idx, launch_kernel1d, CUBuffer, CUDA};

    #[cfg(feature = "vulkan")]
    pub use crate::Vulkan;
}

#[cfg(test)]
pub mod tests_helper {
    use core::ops::Add;

    use crate::{Buffer, Device, Number, Shape, Unit};

    pub trait AddEw<T: Unit, D: Device, S: Shape>: Device {
        fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
    }

    pub fn add_ew_slice<T: Add<Output = T> + Copy>(lhs: &[T], rhs: &[T], out: &mut [T]) {
        for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(out.iter_mut()) {
            *out = *lhs + *rhs;
        }
    }

    pub fn roughly_eq_slices<T: Number>(lhs: &[T], rhs: &[T]) {
        for (a, b) in lhs.iter().zip(rhs) {
            if (a.as_f64() - b.as_f64()).abs() >= 0.1 {
                panic!(
                    "Slices 
                    left {lhs:?} 
                    and right {rhs:?} do not equal. 
                    Encountered diffrent value: {a}, {b}"
                )
            }
        }
    }
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

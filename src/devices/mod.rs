//! This module defines all available compute devices

mod cache2;
pub use cache2::*;

mod generic_blas;
pub use generic_blas::*;

mod addons;
pub use addons::*;

use crate::{shape::Shape, AddGraph, Alloc, Buffer, Device, PtrType};

#[cfg(not(feature = "no-std"))]
pub mod cache;

#[cfg(not(feature = "no-std"))]
#[cfg(feature = "autograd")]
pub(crate) mod borrowing_cache;

//pub mod cache;
#[cfg(not(feature = "no-std"))]
pub use cache::*;

//pub use cache::{Cache, CacheReturn};

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "stack")]
pub mod stack;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "network")]
pub mod network;

mod stack_array;
pub use stack_array::*;

mod cdatatype;
pub use cdatatype::*;

#[cfg(all(any(feature = "cpu", feature = "stack"), feature = "macro"))]
mod cpu_stack_ops;

#[cfg(not(feature = "no-std"))]
mod ident;
#[cfg(not(feature = "no-std"))]
pub use ident::*;

/// Implementors of this trait can be used as cache for a device.
pub trait CacheAble<D: Device> {
    /// May allocate a new buffer or return an existing one.
    /// It may use the cache count provided by the cache count ([Ident]).
    /// This depends on the type of cache.
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
    /// // unsafe, because the next .retrieve call will tehn return the same buffer
    /// unsafe { set_count(0) }
    ///
    /// let buf_2 = device.retrieve::<f32, ()>(10, ());
    ///
    /// assert_eq!(buf.ptr, buf_2.ptr);
    ///
    /// ```
    fn retrieve<T, S: Shape>(device: &D, len: usize, add_node: impl AddGraph) -> Buffer<T, D, S>
    where
        for<'a> D: Alloc<'a, T, S>;

    /// May return an existing buffer using the provided [`Ident`].
    /// This function panics if no buffer with the provided [`Ident`] exists.
    ///
    /// # Safety
    /// This function is unsafe because it is possible to return multiple `Buffer` with `Ident` that share the same memory.
    /// If this function is called twice with the same `Ident`, the returned `Buffer` will be the same.
    /// Even though the return `Buffer`s are owned, this does not lead to double-frees (see [`AllocFlag`]).
    unsafe fn get_existing_buf<T, S: Shape>(device: &D, id: Ident) -> Option<Buffer<T, D, S>>;

    /// Removes a `Buffer` with the provided [`Ident`] from the cache.
    /// This function is internally called when a `Buffer` with [`AllocFlag`] `None` is dropped.
    fn remove(device: &D, ident: Ident);

    /// Adds a pointer that was allocated by [`Alloc`] to the cache and returns a new corresponding [`Ident`].
    /// This function is internally called when a `Buffer` with [`AllocFlag`] `None` is created.
    fn add_to_cache<T, S: Shape>(device: &D, ptr: &D::Ptr<T, S>) -> Ident;
}

// TODO: Mind num implement?
impl<D: Device> CacheAble<D> for () {
    #[inline]
    fn retrieve<T, S: Shape>(device: &D, len: usize, _add_node: impl AddGraph) -> Buffer<T, D, S>
    where
        for<'a> D: Alloc<'a, T, S>,
    {
        Buffer::new(device, len)
    }

    #[inline]
    fn remove(_device: &D, _ident: Ident) {}

    #[inline]
    fn add_to_cache<T, S: Shape>(_device: &D, ptr: &<D as Device>::Ptr<T, S>) -> Ident {
        Ident::new_bumped(ptr.size())
    }

    #[inline]
    unsafe fn get_existing_buf<T, S: Shape>(_device: &D, _id: Ident) -> Option<Buffer<T, D, S>> {
        None
    }
}

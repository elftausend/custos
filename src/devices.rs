//! This module defines all available compute devices

mod generic_blas;
pub use generic_blas::*;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "stack")]
pub mod stack;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "vulkan")]
pub mod vulkan;
// #[cfg(feature = "wgpu")]
// pub mod wgpu;

#[cfg(feature = "network")]
pub mod network;

#[cfg(feature = "nnapi")]
pub mod nnapi;

mod stack_array;
pub use stack_array::*;

mod cdatatype;
pub use cdatatype::*;

#[cfg(any(feature = "cpu", feature = "stack"))]
pub mod cpu_stack_ops;

use crate::{Buffer, HasId, OnDropBuffer, PtrType, ShallowCopy, Shape};

pub trait Device: OnDropBuffer + Sized {
    type Base<T, S: Shape>: HasId + PtrType;
    type Data<T, S: Shape>: HasId + PtrType;

    type Error;

    #[inline]
    fn new() -> Result<Self, Self::Error> {
        todo!()
    }

    // add default impl if GAT default go stable
    // FIXME: probably a better way to realize these
    fn base_to_data<T, S: Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S>;
    fn wrap_to_data<T, S: Shape>(&self, wrap: Self::Wrap<T, Self::Base<T, S>>) -> Self::Data<T, S>;
    fn data_as_wrap<'a, T, S: Shape>(
        data: &'a Self::Data<T, S>,
    ) -> &'a Self::Wrap<T, Self::Base<T, S>>;
    fn data_as_wrap_mut<'a, T, S: Shape>(
        data: &'a mut Self::Data<T, S>,
    ) -> &'a mut Self::Wrap<T, Self::Base<T, S>>;

    /// Creates a new [`Buffer`] using `A`.
    ///
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Device, Base};
    ///
    /// let device = CPU::<Base>::new();
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
}

#[macro_export]
macro_rules! impl_buffer_hook_traits {
    ($device:ident) => {
        impl<T, D: Device, S: Shape, Mods: $crate::OnNewBuffer<T, D, S>>
            $crate::OnNewBuffer<T, D, S> for $device<Mods>
        {
            #[inline]
            fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
                self.modules.on_new_buffer(device, new_buf)
            }
        }

        impl<Mods: $crate::OnDropBuffer> $crate::OnDropBuffer for $device<Mods> {
            #[inline]
            fn on_drop_buffer<T, D: Device, S: Shape>(&self, device: &D, buf: &Buffer<T, D, S>) {
                self.modules.on_drop_buffer(device, buf)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_device_traits {
    ($device:ident) => {
        $crate::impl_retriever!($device);
        $crate::impl_buffer_hook_traits!($device);
        $crate::impl_wrapped_data!($device);

        #[cfg(feature = "graph")]
        crate::pass_down_optimize_mem_graph!($device);

        $crate::pass_down_grad_fn!($device);
        $crate::pass_down_tape_actions!($device);

        $crate::pass_down_replace_buf_dev!($device);
    };
}

#[macro_export]
macro_rules! impl_retriever {
    ($device:ident, $($trait_bounds:tt)*) => {
        impl<T: $( $trait_bounds )*, Mods: $crate::Retrieve<Self, T, S>, S: $crate::Shape> $crate::Retriever<T, S> for $device<Mods> {
            #[inline]
            fn retrieve<const NUM_PARENTS: usize>(
                &self,
                len: usize,
                parents: impl $crate::Parents<NUM_PARENTS>,
            ) -> Buffer<T, Self, S> {
                let data = unsafe { self
                    .modules
                    .retrieve::<NUM_PARENTS>(self, len, parents) };
                let buf = Buffer {
                    data,
                    device: Some(self),
                };
                self.modules.on_retrieve_finish(&buf);
                buf
            }
        }
    };

    ($device:ident) => {
        $crate::impl_retriever!($device, Sized);
    }
}

mod cpu;
pub use cpu::*;

mod cuda;
pub use cuda::*;

use super::{Alloc, OnDropBuffer};

pub trait Device: Alloc + OnDropBuffer {
    type Error;

    #[inline]
    fn new() -> Result<Self, Self::Error> {
        todo!()
    }
}

#[macro_export]
macro_rules! impl_buffer_hook_traits {
    ($device:ident) => {
        impl<T, D: Device, S: Shape, Mods: OnNewBuffer<T, D, S>> OnNewBuffer<T, D, S>
            for $device<Mods>
        {
            #[inline]
            fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
                self.modules.on_new_buffer(device, new_buf)
            }
        }

        impl<Mods: OnDropBuffer> OnDropBuffer for $device<Mods> {
            #[inline]
            fn on_drop_buffer<'a, T, D: Device, S: Shape>(
                &self,
                device: &'a D,
                buf: &Buffer<T, D, S>,
            ) {
                self.modules.on_drop_buffer(device, buf)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_retriever {
    ($device:ident) => {
        impl<Mods: Retrieve<Self>> Retriever for $device<Mods> {
            #[inline]
            fn retrieve<T: 'static, S: Shape>(&self, len: usize) -> Buffer<T, Self, S> {
                let data = self.modules.retrieve::<T, S>(self, len);
                let buf = Buffer {
                    data,
                    device: Some(self),
                };
                self.modules.on_retrieve_finish(&buf);
                buf
            }
        }
    };
}

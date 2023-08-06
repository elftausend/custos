use crate::{module_comb::{LazySetup, Buffer, OnDropBuffer, Alloc}, Shape, cuda::CUDAPtr};

use super::Device;

pub trait IsCuda {}

pub struct CUDA<Mods> {
    modules: Mods,
}

impl<Mods: OnDropBuffer> Device for CUDA<Mods> {
    type Error = i32;
}

impl<Mods: OnDropBuffer> OnDropBuffer for CUDA<Mods> {
    #[inline]
    fn on_drop_buffer<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<Mods> Alloc for CUDA<Mods> {
    type Data<T, S: Shape> = CUDAPtr<T>;

    fn alloc<T, S: Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Data<T, S> {
        todo!()
    }

    fn alloc_from_slice<T, S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone {
        todo!()
    }

    
}

impl<Mods> IsCuda for CUDA<Mods> {}

impl<Mods> LazySetup for CUDA<Mods> {
    #[inline]
    fn lazy_setup(&mut self) {
        // switch to stream record mode for graph
    }
}

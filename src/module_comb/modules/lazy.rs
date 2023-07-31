use core::marker::PhantomData;

use crate::{
    module_comb::{Alloc, Buffer, Device, Module, OnDropBuffer, Retrieve, Setup},
    Shape,
};

#[derive(Debug, Default)]
pub struct Lazy<Mods> {
    mods: Mods,
}

pub trait LazySetup {
    fn lazy_setup(&mut self) {}
}

impl<Mods: Module<D>, D: LazySetup> Module<D> for Lazy<Mods> {
    type Module = Lazy<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        Lazy { mods: Mods::new() }
    }
}

impl<D: LazySetup, Mods: Setup<D>> Setup<D> for Lazy<Mods> {
    #[inline]
    fn setup(device: &mut D) {
        device.lazy_setup();
        println!("setup lazy");
        Mods::setup(device)
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Lazy<Mods> {
    #[inline]
    fn on_drop_buffer<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        self.mods.on_drop_buffer(device, buf)
    }
}

impl<Mods: OnDropBuffer, D> Retrieve<D> for Lazy<Mods> {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: crate::module_comb::Alloc,
    {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::module_comb::{Alloc, Base, CPU, CUDA};

    use super::Lazy;

    #[test]
    fn test_lazy_device_use() {
        // let device = CPU::<Lazy<Base>>::new();
        // let data = device.alloc::<f32, ()>(10, crate::flag::AllocFlag::None);
    }

    #[test]
    fn test_lazy_device_use_cuda() {
        // let device = CUDA::<Lazy<Base>>::new();
        // let data = device.alloc::<f32, ()>(10, crate::flag::AllocFlag::None);
    }
}

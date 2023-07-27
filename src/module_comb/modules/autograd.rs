mod gradients;
mod tape;

pub use gradients::*;
pub use tape::*;

use core::marker::PhantomData;

use crate::{
    module_comb::{Alloc, Buffer, HasId, Module, OnNewBuffer, Retrieve, Setup},
    Shape,
};

#[derive(Debug, Default)]
pub struct Autograd<Mods> {
    // AutogradModule is not needed -> remove PhantomData
    pd: PhantomData<Mods>,
    grads: Gradients,
}

impl<Mods> OnNewBuffer for Autograd<Mods> {
    #[inline]
    fn on_new_buffer<T, S, D>(&self, _device: &D, new_buf: &Buffer<T, D, S>)
    where
        S: Shape,
        D: Alloc,
        D::Data<T, S>: HasId,
    {
        self.grads
            .no_grads_pool
            .borrow_mut()
            .cache
            .insert(*new_buf.id(), Box::new(new_buf));
    }
}

impl<Mods: Module<SD>, SD: Alloc> Module<SD> for Autograd<Mods> {
    type Module = AutogradModule<Mods::Module, SD>;

    #[inline]
    fn new() -> Self::Module {
        AutogradModule {
            modules: Mods::new(),
            pd: PhantomData,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct AutogradModule<Mods, D: Alloc> {
    modules: Mods,
    pd: PhantomData<D>,
}

impl<Mods: Setup<NewDev>, D: Alloc, NewDev> Setup<NewDev> for AutogradModule<Mods, D> {
    #[inline]
    fn setup(device: &mut NewDev) {
        Mods::setup(device)
    }
}

impl<Mods: Retrieve<D>, D, SimpleDevice: Alloc> Retrieve<D> for AutogradModule<Mods, SimpleDevice> {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc,
    {
        self.modules.retrieve(device, len)
    }
}

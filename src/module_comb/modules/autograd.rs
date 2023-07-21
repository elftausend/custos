use core::marker::PhantomData;

use crate::module_comb::{Alloc, Module, Retrieve};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Autograd<Mods> {
    pd: PhantomData<Mods>,
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

impl<Mods: Retrieve<D>, D, SimpleDevice: Alloc> Retrieve<D> for AutogradModule<Mods, SimpleDevice> {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc,
    {
        self.modules.retrieve(device, len)
    }
}

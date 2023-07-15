use crate::module_comb::{Alloc, Retrieve, Module};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Autograd<Mods> {
    modules: Mods,
}

impl<Mods: Default, D> Module<D> for Autograd<Mods> {
    type Module = Autograd<Mods>;

    fn new() -> Self::Module {
        Default::default()
    }
}

impl<Mods: Retrieve<D>, D> Retrieve<D> for Autograd<Mods> {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc,
    {
        self.modules.retrieve(device, len)
    }
}

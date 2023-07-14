use crate::feature_comb::{Alloc, Retrieve};

pub struct Autograd<Mods> {
    modules: Mods,
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

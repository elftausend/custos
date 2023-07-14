use crate::feature_comb::{Alloc, Retrieve};

pub struct Cache<D: Alloc> {
    device: D::Data<u8, ()>
}

pub struct Cached<Mods> {
    modules: Mods,
}

impl<Mods, D> Retrieve<D> for Cached<Mods> {
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> D::Data<T, S>
    where
        D: Alloc,
    {
        todo!()
    }
}

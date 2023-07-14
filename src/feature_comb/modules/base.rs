use crate::{
    feature_comb::{Alloc, Retrieve},
    flag::AllocFlag,
};

pub struct Base;

impl<D> Retrieve<D> for Base {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc,
    {
        device.alloc(len, AllocFlag::None)
    }
}

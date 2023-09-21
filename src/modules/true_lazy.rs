use crate::{Device, Shape, Retrieve, Buffer, OnDropBuffer, impl_buffer_hook_traits};

pub struct TrueLazy<Mods> {
    modules: Mods
}

impl_buffer_hook_traits!(TrueLazy);

impl<Mods: OnDropBuffer, T, D, S: Shape> Retrieve<D, T, S> for TrueLazy<Mods>
where
    D: Device,
    D::Data<T, S>:
{
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
    ) -> <D>::Data<T, S>
    where
        S: Shape,
        D: Device + crate::Alloc<T> 
    {
        todo!()
    }
}
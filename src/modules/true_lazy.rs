use core::panic::Location;

use crate::{flag::AllocFlag, Buffer, Device, HashLocation, Retrieve, Shape};

pub struct TrueLazy<Mods> {
    modules: Mods,
}

// impl_buffer_hook_traits!(TrueLazy);

impl<Mods, T, D, S: Shape> Retrieve<D, T, S> for TrueLazy<Mods>
where
    D: Device,
    D::Data<T, S>: Default,
{
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        parents: impl crate::Parents<NUM_PARENTS>,
        alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> <D>::Data<T, S>
    where
        S: Shape,
        D: Device + crate::Alloc<T>,
    {
        // TODO: alloc with onion for new buffer (return dangling)

        let location: HashLocation = Location::caller().into();

        D::Data::<T, S>::default()
    }
}

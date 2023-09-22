use core::cell::Cell;

use crate::{flag::AllocFlag, Device, Retrieve, Shape, HasId, Module, LazySetup, Setup};

pub struct TrueLazy<Mods> {
    modules: Mods,
    id: Cell<u64>
}

impl<Mods: Module<D>, D: LazySetup> Module<D> for TrueLazy<Mods> {
    type Module = TrueLazy<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        TrueLazy {
            modules: Mods::new(),
            id: Cell::default()
        }
    }
}

impl<Mods, D> Setup<D> for TrueLazy<Mods> {}

// impl_buffer_hook_traits!(TrueLazy);

impl<Mods, T, D, S: Shape> Retrieve<D, T, S> for TrueLazy<Mods>
where
    D: Device,
    D::Data<T, S>: Default + HasId
{
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
        alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> <D>::Data<T, S>
    where
        S: Shape,
        D: Device + crate::Alloc<T>,
    {
        let mut data = D::Data::<T, S>::default();
        let id = self.id.take();
        unsafe {
            data.set_id(id)
        }
        self.id.set(id + 1);
        data
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    #[test]
    fn test_true_lazy_id_set() {
        use crate::{CPU, TrueLazy, Base, Buffer, HasId, Id};

        let device = CPU::<TrueLazy<Base>>::new();
        let buf = Buffer::<i32, _>::new(&device, 10);
        assert_eq!(buf.id(), Id { id: 0, len: 10 });
        let buf = Buffer::<i32, _>::new(&device, 10);
    }
}

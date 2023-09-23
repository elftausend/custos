mod lazy_graph;
mod ty;

use crate::{
    flag::AllocFlag, AddOperation, Device, HasId, LazySetup, Module, PtrType, Retrieve, Setup,
    Shape, PtrConv, MainMemory,
};
use core::cell::Cell;
use lazy_graph::LazyGraph;

pub struct TrueLazy<Mods> {
    modules: Mods,
    graph: LazyGraph,
    id: Cell<u64>,
}

impl<Mods: Module<D>, D: LazySetup> Module<D> for TrueLazy<Mods> {
    type Module = TrueLazy<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        TrueLazy {
            modules: Mods::new(),
            graph: LazyGraph::default(),
            id: Cell::default(),
        }
    }
}

impl<Mods, D> Setup<D> for TrueLazy<Mods> {}

// impl_buffer_hook_traits!(TrueLazy);

impl<Mods, T, D, S: Shape> Retrieve<D, T, S> for TrueLazy<Mods>
where
    D: Device,
    D::Data<T, S>: Default + HasId,
{
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        _device: &D,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
        _alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> <D>::Data<T, S>
    where
        S: Shape,
        D: Device + crate::Alloc<T>,
    {
        let mut data = D::Data::<T, S>::default();
        let id = self.id.take();
        unsafe {
            data.set_id(id);
            data.set_size(len)
        }
        self.id.set(id + 1);
        data
    }
}

impl<Mods, D: Device + PtrConv> AddOperation<D> for TrueLazy<Mods> {
    fn add_operation2<T, S: Shape>(
        &self,
        out: &mut crate::Buffer<T, D, S>,
        operation: impl Fn(&mut crate::Buffer<T, D, S>),
    ) {
        self.graph.add_operation(operation)    
    }

    fn call_lazily(&self) {}
}

#[cfg(feature = "autograd")]
impl<Mods: crate::TapeActions> crate::TapeActions for TrueLazy<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<super::Tape>> {
        self.modules.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<super::Tape>> {
        self.modules.tape_mut()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    #[test]
    fn test_true_lazy_id_set() {
        use crate::{Base, Buffer, HasId, Id, TrueLazy, CPU};

        let device = CPU::<TrueLazy<Base>>::new();
        let buf = Buffer::<i32, _>::new(&device, 10);
        assert_eq!(buf.id(), Id { id: 0, len: 10 });
        let buf = Buffer::<i32, _>::new(&device, 10);
        assert_eq!(buf.id(), Id { id: 1, len: 10 });
    }
}

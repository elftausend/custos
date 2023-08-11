use crate::{
    flag::AllocFlag,
    module_comb::{
        AddOperation, Alloc, Device, Module, OnDropBuffer, OnNewBuffer, Parents, Retrieve, Setup,
        TapeActions,
    },
    Shape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Base;

impl<D> Module<D> for Base {
    type Module = Base;

    #[inline]
    fn new() -> Self::Module {
        Base
    }
}

impl AddOperation for Base {
    #[inline]
    fn add_operation(&self, mut operation: impl FnOnce()) {
        operation();
    }
}

impl<D> Setup<D> for Base {}

impl<T, D: Device, S: Shape> OnNewBuffer<T, D, S> for Base {}

impl OnDropBuffer for Base {}

impl<D> Retrieve<D> for Base {
    #[inline]
    fn retrieve<T, S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        _parents: impl Parents<NUM_PARENTS>,
    ) -> <D>::Data<T, S>
    where
        S: crate::Shape,
        D: Alloc,
    {
        device.alloc(len, AllocFlag::None)
    }
}

impl TapeActions for Base {}

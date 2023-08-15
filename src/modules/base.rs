use core::any::Any;

use crate::{
    flag::AllocFlag, AddOperation, Alloc, Buffer, Device, Module, OnDropBuffer, OnNewBuffer,
    Parents, Retrieve, Setup, Shape, TapeActions,
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
    unsafe fn add_operation<T: 'static, D: Device + 'static, S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut dyn Any),
    ) {
        let out: &mut Buffer<T, D, S> = unsafe { std::mem::transmute(out) };
        operation(out);
    }

    #[inline]
    fn add_operation2(&self, mut operation: impl crate::Operation) {
        operation.forward()
    }
}

impl<D> Setup<D> for Base {}

impl<T, D: Device, S: Shape> OnNewBuffer<T, D, S> for Base {}

impl OnDropBuffer for Base {}

impl<D, T> Retrieve<D, T> for Base {
    #[inline]
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        _parents: impl Parents<NUM_PARENTS>,
    ) -> <D>::Data<T, S>
    where
        S: crate::Shape,
        D: Alloc<T>,
    {
        device.alloc(len, AllocFlag::None)
    }
}

impl TapeActions for Base {}

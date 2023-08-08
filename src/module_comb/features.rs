use core::cell::{Ref, RefMut};

use crate::Shape;

use super::{Alloc, Buffer, Device, GradFn, Gradients, Id, OnDropBuffer, Tape};

pub trait Feature: OnDropBuffer {}

// is a cached module is placed before Autograd results a problem
// -> the retrieved buffer is not added to the no grads pool of the autograd module
// let device = CPU::<Cached<Autograd<Base>>>::new();
//
// how to fix this:
// add retrieved buffer to no grads pool at the end of the chain (at device level (Retriever trait))
// => "generator", "actor"
pub trait Retrieve<D>: OnDropBuffer {
    // "generator"
    #[track_caller]
    fn retrieve<T, S: Shape>(&self, device: &D, len: usize) -> D::Data<T, S>
    where
        T: 'static, // if 'static causes any problems -> put T to => Retrieve<D, T>?
        D: Alloc;

    // "actor"
    #[inline]
    fn on_retrieve_finish<T, S: Shape>(&self, _retrieved_buf: &Buffer<T, D, S>)
    where
        T: 'static,
        D: Device,
    {
    }
}

pub trait HasModules<Mods> {
    fn modules(&self) -> &Mods;
}

pub trait TapeActions {
    // "generator" - do not forget to pass down
    #[inline]
    fn tape(&self) -> Option<Ref<Tape>> {
        None
    }
    // "generator" - do not forget to pass down
    #[inline]
    fn tape_mut(&self) -> Option<RefMut<Tape>> {
        None
    }

    // use track caller to identify a specific grad function
    //-> if backward is not called (.drain()), the grad fn vector will gradually fill up
    #[track_caller]
    fn add_grad_fn<T, S: Shape, const N: usize>(
        &self,
        ids: impl AllocGradsFrom<N>,
        grad_fn: impl Fn(&mut Gradients) + 'static,
    ) where
        T: 'static,
        Self: Device + 'static,
    {
        if let Some(mut tape) = self.tape_mut() {
            // the type T must match for every Id!
            for id in ids.ids() {
                tape.grads.grads_pool.add_buf_once::<T, Self, S>(self, id)
            }

            tape.add_grad_fn(grad_fn)
        }
    }
}

pub trait AllocGradsFrom<const N: usize> {
    fn ids(self) -> [Id; N];
}

impl AllocGradsFrom<1> for Id {
    #[inline]
    fn ids(self) -> [Id; 1] {
        [self]
    }
}

impl AllocGradsFrom<2> for (Id, Id) {
    #[inline]
    fn ids(self) -> [Id; 2] {
        [self.0, self.1]
    }
}

impl AllocGradsFrom<3> for (Id, Id, Id) {
    #[inline]
    fn ids(self) -> [Id; 3] {
        [self.0, self.1, self.2]
    }
}

impl<const N: usize> AllocGradsFrom<N> for [Id; N] {
    #[inline]
    fn ids(self) -> [Id; N] {
        self
    }
}

use core::cell::{Ref, RefMut};

use crate::Shape;

use super::{Alloc, Buffer, Device, GradFn, Gradients, Id, OnDropBuffer, Tape, HasId};

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
    fn retrieve<T, S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> D::Data<T, S>
    where
        T: 'static, // if 'static causes any problems -> put T to => Retrieve<D, T>?
        S: Shape,
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
    fn add_grad_fn<T, S: Shape>(
        &self,
        // ids: impl AllocGradsFrom<N>,
        grad_fn: impl Fn(&mut Gradients) + 'static,
    ) where
        T: 'static,
        Self: Device + 'static,
    {
        if let Some(mut tape) = self.tape_mut() {
            // the type T must match for every Id!
            // for id in ids.ids() {
            //     tape.grads.grads_pool.add_buf_once::<T, Self, S>(self, id)
            // }

            tape.add_grad_fn(grad_fn)
        }
    }
}

pub trait Parents<const N: usize> {
    fn ids(self) -> [Id; N];
}

impl Parents<0> for () {
    #[inline]
    fn ids(self) -> [Id; 0] {
        []
    }
}

impl Parents<1> for Id {
    #[inline]
    fn ids(self) -> [Id; 1] {
        [self]
    }
}

impl Parents<2> for (Id, Id) {
    #[inline]
    fn ids(self) -> [Id; 2] {
        [self.0, self.1]
    }
}

impl Parents<3> for (Id, Id, Id) {
    #[inline]
    fn ids(self) -> [Id; 3] {
        [self.0, self.1, self.2]
    }
}

impl<const N: usize> Parents<N> for [Id; N] {
    #[inline]
    fn ids(self) -> [Id; N] {
        self
    }
}

impl<T, D: Device, S: Shape> Parents<1> for &Buffer<'_, T, D, S> {
    #[inline]
    fn ids(self) -> [Id; 1] {
        [self.id()]
    }
}

impl<T, D: Device, S: Shape, T1, D1: Device, S1: Shape> Parents<2> for (&Buffer<'_, T, D, S>, &Buffer<'_, T1, D1, S1>) {
    #[inline]
    fn ids(self) -> [Id; 2] {
        let (lhs, rhs) = self;
        [lhs.id(), rhs.id()]
    }
}

impl<T, D: Device, S: Shape, T1, D1: Device, S1: Shape, T2, D2: Device, S2: Shape> Parents<3> for (&Buffer<'_, T, D, S>, &Buffer<'_, T1, D1, S1>, &Buffer<'_, T2, D2, S2>) {
    #[inline]
    fn ids(self) -> [Id; 3] {
        let (buf, buf1, buf2) = self;
        [buf.id(), buf1.id(), buf2.id()]
    }
}

impl<T, D: Device, S: Shape, const N: usize> Parents<N> for [&Buffer<'_, T, D, S>; N] {
    #[inline]
    fn ids(self) -> [Id; N] {
        self.map(|buf| buf.id())
    }
}

pub trait AddOperation {
    fn add_operation(&self, operation: impl FnOnce());
    fn call_lazily(&self) {}
}

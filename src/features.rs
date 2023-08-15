use core::{
    any::Any,
    cell::{Ref, RefMut},
};

use crate::{Parents, Shape};

use super::{Alloc, Buffer, Device, OnDropBuffer};

pub trait Feature: OnDropBuffer {}

// is a cached module is placed before Autograd results a problem
// -> the retrieved buffer is not added to the no grads pool of the autograd module
// let device = CPU::<Cached<Autograd<Base>>>::new();
//
// how to fix this:
// add retrieved buffer to no grads pool at the end of the chain (at device level (Retriever trait))
// => "generator", "actor"
pub trait Retrieve<D, T>: OnDropBuffer {
    // "generator"
    #[track_caller]
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> D::Data<T, S>
    where
        S: Shape,
        D: Device + Alloc<T>;

    // "actor"
    #[inline]
    fn on_retrieve_finish<S: Shape>(&self, _retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
    }
}

pub trait HasModules<Mods> {
    fn modules(&self) -> &Mods;
}

pub trait TapeActions {
    // "generator" - do not forget to pass down
    #[inline]
    fn tape(&self) -> Option<Ref<crate::Tape>> {
        None
    }
    // "generator" - do not forget to pass down
    #[inline]
    fn tape_mut(&self) -> Option<RefMut<crate::Tape>> {
        None
    }

    // use track caller to identify a specific grad function
    //-> if backward is not called (.drain()), the grad fn vector will gradually fill up
    #[track_caller]
    fn add_grad_fn<T, S: Shape>(
        &self,
        // ids: impl AllocGradsFrom<N>,
        grad_fn: impl Fn(&mut crate::Gradients) + 'static,
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

pub trait Operation {
    fn forward(&mut self);
}

pub trait AddOperation {
    fn add_operation2(&self, _operation: impl Operation) {}
    unsafe fn add_operation<T: 'static, D: Device + 'static, S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut dyn Any),
    );
    fn call_lazily(&self) {}
}

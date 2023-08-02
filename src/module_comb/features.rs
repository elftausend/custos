use crate::Shape;

use super::{Alloc, GradFn, OnDropBuffer, Buffer, Device};

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
        D: Device {}
}

pub trait HasModules<Mods> {
    fn modules(&self) -> &Mods;
}

pub trait AddGradFn<D> {
    fn add_grad_fn(&self, device: &D, grad_fn: GradFn);
}

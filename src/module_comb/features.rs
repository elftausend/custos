use crate::Shape;

use super::{Alloc, GradFn, OnDropBuffer};

pub trait Feature: OnDropBuffer {}

pub trait Retrieve<D>: OnDropBuffer {
    #[track_caller]
    fn retrieve<T, S: Shape>(&self, device: &D, len: usize) -> D::Data<T, S>
    where
        T: 'static, // if 'static causes any problems -> put T to => Retrieve<D, T>?
        D: Alloc;
}

pub trait HasModules<Mods> {
    fn modules(&self) -> &Mods;
}

pub trait AddGradFn<D> {
    fn add_grad_fn(&self, device: &D, grad_fn: GradFn);
}

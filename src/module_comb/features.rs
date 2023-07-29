use core::marker::PhantomData;

use crate::Shape;

use super::{Alloc, Buffer, Device, Gradients, HasId, OnDropBuffer};

pub trait Feature: OnDropBuffer {}

pub trait Retrieve<D>: OnDropBuffer {
    #[track_caller]
    fn retrieve<T, S: Shape>(&self, device: &D, len: usize) -> D::Data<T, S>
    where
        D: Alloc;
}

pub trait HasModules<Mods> {
    fn modules(&self) -> &Mods;
}

// does not require the device param ???
type GradFn<D> = Box<dyn Fn(&mut Gradients, &D)>;

pub trait AddGradFn<D> {
    fn add_grad_fn(&self, device: &D, grad_fn: GradFn<D>);
}

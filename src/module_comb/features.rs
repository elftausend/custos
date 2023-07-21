use core::marker::PhantomData;

use crate::Shape;

use super::{Alloc, Device};

pub trait Retrieve<D> {
    #[track_caller]
    fn retrieve<T, S: Shape>(&self, device: &D, len: usize) -> D::Data<T, S>
    where
        D: Alloc;
}

pub struct Gradients<D> {
    pd: PhantomData<D>,
}

type GradFn<D> = Box<dyn Fn(&mut Gradients<D>, &D)>;

pub trait AddGradFn<D> {
    fn add_grad_fn(&self, device: &D, grad_fn: GradFn<D>);
}

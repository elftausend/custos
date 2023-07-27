use core::marker::PhantomData;

use crate::Shape;

use super::{Alloc, Buffer, Device, Gradients, HasId};

pub trait Retrieve<D> {
    #[track_caller]
    fn retrieve<T, S: Shape>(&self, device: &D, len: usize) -> D::Data<T, S>
    where
        D: Alloc;
}

pub trait HasModules<Mods> {
    fn modules(&self) -> &Mods;
}

pub trait OnNewBuffer {
    fn on_new_buffer<T, S, D>(&self, _device: &D, _new_buf: &Buffer<T, D, S>)
    where
        D::Data<T, S>: HasId,
        S: Shape,
        D: Alloc,
    {
    }
}

// does not require the device param ???
type GradFn<D> = Box<dyn Fn(&mut Gradients, &D)>;

pub trait AddGradFn<D> {
    fn add_grad_fn(&self, device: &D, grad_fn: GradFn<D>);
}

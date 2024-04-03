use crate::{ApplyFunction, Retrieve, Shape};

use super::untyped_device::Untyped;


impl<Mods: Retrieve<Self, T, S>, T, S: Shape> ApplyFunction<T, S> for Untyped<Mods> {
    fn apply_fn<F>(
        &self,
        // buf: &D::Data<T, S>,
        buf: &crate::Buffer<T, Self, S>,
        f: impl Fn(crate::Resolve<T>) -> F + Copy + 'static,
    ) -> crate::Buffer<T, Self, S>
    where
        F: crate::TwoWay<T> + 'static 
    {
        todo!()
    }
}


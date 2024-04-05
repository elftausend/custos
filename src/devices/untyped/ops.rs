use crate::{ApplyFunction, Retrieve, Shape};

use super::{untyped_device::Untyped, AsType, MatchesType};


impl<Mods: Retrieve<Self, T, S>, T: AsType, S: Shape> ApplyFunction<T, S> for Untyped<Mods> {
    fn apply_fn<F>(
        &self,
        // buf: &D::Data<T, S>,
        buf: &crate::Buffer<T, Self, S>,
        f: impl Fn(crate::Resolve<T>) -> F + Copy + 'static,
    ) -> crate::Buffer<T, Self, S>
    where
        F: crate::TwoWay<T> + 'static 
    {
        let res = buf.base();
        buf.base().matches_storage_type::<T>().unwrap();
        todo!()
    }
}


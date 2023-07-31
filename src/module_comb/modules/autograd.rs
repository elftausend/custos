mod gradients;
mod tape;

pub use gradients::*;
pub use tape::*;

use core::{any::Any, hash::BuildHasher, mem::transmute};
use std::collections::HashMap;

use crate::{
    module_comb::{
        Alloc, Buffer, Device, HasId, Id, Module, OnDropBuffer, OnNewBuffer, Retrieve, Setup,
        UniqueId,
    },
    Shape,
};

#[derive(Debug, Default)]
pub struct Autograd<Mods> {
    modules: Mods,
    grads: Gradients,
}

#[inline]
pub unsafe fn register_buf<'a, T, D, S>(
    cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>,
    buf: &'a Buffer<T, D, S>,
) where
    T: 'static,
    D: Device + 'static,
    S: Shape,
{
    let buf: &'static Buffer<T, D, S> = transmute(buf);
    cache.insert(*buf.id(), Box::new(buf));
}

#[inline]
pub fn unregister_buf(cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>, id: Id) {
    cache.remove(&id);
}

impl<T, D, S, Mods> OnNewBuffer<T, D, S> for Autograd<Mods>
where
    T: 'static,
    D: Device + 'static,
    S: Shape,
    Mods: OnNewBuffer<T, D, S>,
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        unsafe { register_buf(&mut self.grads.no_grads_pool.borrow_mut().cache, new_buf) };

        // pass down
        self.modules.on_new_buffer(device, new_buf)
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Autograd<Mods> {
    #[inline]
    fn on_drop_buffer<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        unregister_buf(&mut self.grads.no_grads_pool.borrow_mut().cache, buf.id());
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<Mods: Module<D>, D> Module<D> for Autograd<Mods> {
    type Module = Autograd<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        Autograd {
            modules: Mods::new(),
            grads: Gradients::default(),
        }
    }
}

impl<Mods: Setup<NewDev>, NewDev> Setup<NewDev> for Autograd<Mods> {
    #[inline]
    fn setup(device: &mut NewDev) {
        Mods::setup(device)
    }
}

impl<Mods: Retrieve<D>, D> Retrieve<D> for Autograd<Mods> {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc,
    {
        self.modules.retrieve(device, len)
    }
}

/*
impl<Mods: Module<SD>, SD: Alloc> Module<SD> for Autograd<Mods> {
    type Module = AutogradModule<Mods::Module, SD>;

    #[inline]
    fn new() -> Self::Module {
        AutogradModule {
            modules: Mods::new(),
            pd: PhantomData,
        }
    }
}

// remove
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct AutogradModule<Mods, D: Alloc> {
    modules: Mods,
    pd: PhantomData<D>,
}

impl<Mods: Setup<NewDev>, D: Alloc, NewDev> Setup<NewDev> for AutogradModule<Mods, D> {
    #[inline]
    fn setup(device: &mut NewDev) {
        Mods::setup(device)
    }
}

impl<Mods: OnDropBuffer, SD: Alloc> OnDropBuffer for AutogradModule<Mods, SD> {
    fn on_drop<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        self.modules.on_drop(device, buf)
    }
}

impl<Mods: Retrieve<D>, D, SimpleDevice: Alloc> Retrieve<D> for AutogradModule<Mods, SimpleDevice> {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc,
    {
        self.modules.retrieve(device, len)
    }
}
*/
#[cfg(test)]
mod tests {
    use core::any::Any;

    use crate::{
        module_comb::{Base, BorrowCache, Buffer, Cached, Device, HasId, Id, CPU},
        Shape,
    };

    use super::Autograd;

    #[inline]
    pub fn downcast_val<'a, 'b, T: 'static, D: Device + 'static, S: Shape>(
        buf_any: &'b dyn Any,
        _device: &'a D,
    ) -> Option<&'b Buffer<'a, T, D, S>> {
        buf_any.downcast_ref()
    }

    pub fn get_buf_with_dev<'a, 'b, T, D, S>(
        _device: &'a D,
        no_grads_pool: &'b BorrowCache,
        id: Id,
    ) -> Option<&'b Buffer<'a, T, D, S>>
    where
        T: 'static,
        D: Device + 'static,
        S: Shape + 'static,
    {
        no_grads_pool.get_buf(id)
    }

    #[test]
    fn test_buffer_creation_autograd_register() {
        let device = CPU::<Cached<Autograd<Base>>>::new();
        let buf: Buffer<f32, _> = Buffer::<f32, _>::new(&device, 10);
        let autograd = &device.modules.modules;
        {
            let no_grads_pool = &autograd.grads.no_grads_pool.borrow();
            get_buf_with_dev::<f32, _, ()>(&device, no_grads_pool, buf.id()).unwrap();

            /*no_grads_pool.get_buf::<f32, _, ()>(buf.id());
            let buf_any = no_grads_pool.cache.get(&buf.id()).unwrap();
            let buf: &Buffer<f32, _> = downcast_val::<f32, _, _>(buf_any, &device).unwrap();
            println!("no_grads_pool: {no_grads_pool:?}");*/
        }
        drop(buf);
    }
}

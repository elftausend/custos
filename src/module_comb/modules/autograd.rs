mod gradients;
mod tape;

pub use gradients::*;
pub use tape::*;

use core::{any::Any, hash::BuildHasher, marker::PhantomData, mem::transmute};
use std::collections::HashMap;

use crate::{
    module_comb::{Alloc, Buffer, HasId, Id, Module, OnNewBuffer, Retrieve, Setup, UniqueId, Device, OnDropBuffer},
    Shape,
};

#[derive(Debug, Default)]
pub struct Autograd<Mods> {
    // AutogradModule is not needed -> remove PhantomData
    mods: Mods,
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
    Mods: OnNewBuffer<T, D, S>
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        unsafe { register_buf(&mut self.grads.no_grads_pool.borrow_mut().cache, new_buf) };
        
        // pass down
        self.mods.on_new_buffer(device, new_buf)
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Autograd<Mods> {
    #[inline]
    fn on_drop<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        unregister_buf(&mut self.grads.no_grads_pool.borrow_mut().cache, buf.id());
        self.mods.on_drop(device, buf)
    }
}


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

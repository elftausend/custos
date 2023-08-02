mod gradients;
mod tape;

pub use gradients::*;
pub use tape::*;

use core::{any::Any, hash::BuildHasher, mem::transmute};
use std::collections::HashMap;

use crate::{
    flag::AllocFlag,
    module_comb::{
        Alloc, Buffer, Device, HasId, Id, Module, OnDropBuffer, OnNewBuffer, PtrConv, Retrieve,
        Setup, UniqueId,
    },
    Shape,
};

use super::{CachedModule, Cached};

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
    D: Device + PtrConv + 'static,
    S: Shape,
{
    let wrapped_data = D::convert::<T, S, T, S>(&buf.data, AllocFlag::Wrapper);
    let buf = Buffer {
        data: wrapped_data,
        device: buf.device,
    };
    let buf: Buffer<'static, T, D, S> = transmute(buf);
    cache.insert(*buf.id(), Box::new(buf));
}

#[inline]
pub fn unregister_buf(cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>, id: Id) {
    cache.remove(&id);
}

impl<Mods> Autograd<Mods> {
    #[inline]
    pub fn register_no_grad_buf<T, D, S>(&self, buf: &Buffer<T, D, S>)
    where
        T: 'static,
        D: Device + PtrConv + 'static,
        S: Shape,
    {
        let no_grads_pool = &mut self.grads.no_grads_pool.borrow_mut().cache;

        if no_grads_pool.get(&buf.id()).is_some() {
            return;
        }

        unsafe { register_buf(no_grads_pool, buf) };
    }
}

impl<T, D, S, Mods> OnNewBuffer<T, D, S> for Autograd<Mods>
where
    T: 'static,
    D: Device + PtrConv + 'static,
    S: Shape,
    Mods: OnNewBuffer<T, D, S>,
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        self.register_no_grad_buf(new_buf);

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

impl<Mods: Module<D>, D: Alloc> Module<D> for Autograd<Mods> {
    type Module = Autograd<CachedModule<Mods::Module, D>>;

    #[inline]
    fn new() -> Self::Module {
        
        Autograd {
            modules: Cached::<Mods>::new(),
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

impl<Mods: Retrieve<D>, D> Retrieve<D> for Autograd<Mods>
where
    D: PtrConv + Device + 'static,
{
    #[inline]
    fn retrieve<T: 'static, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc,
    {
        let data = self.modules.retrieve(device, len);

        // module specific action: could probably generalize this better
        {
            let buf = Buffer {
                data: unsafe { D::convert::<T, S, T, S>(&data, AllocFlag::Wrapper) },
                device: Some(device),
            };
    
            self.register_no_grad_buf(&buf);
        }
        
        data
    }
}

#[cfg(test)]
mod tests {
    use core::any::Any;

    use crate::{
        module_comb::{Base, Buffer, Cached, Device, HasId, CPU, Retriever},
        Shape,
    };

    use super::Autograd;

    #[inline]
    pub fn downcast_val<'a, 'b, T: 'static, D: Device + 'static, S: Shape>(
        buf_any: &'b Box<dyn Any>,
        _device: &'a D,
    ) -> Option<&'b Buffer<'a, T, D, S>> {
        buf_any.downcast_ref::<Buffer<T, D, S>>()
    }

    #[test]
    fn test_buffer_creation_autograd_register_manual() {
        let device = CPU::<Autograd<Base>>::new();
        let buf: Buffer<f32, _> = Buffer::<f32, _>::new(&device, 10);

        let autograd = &device.modules;
        {
            let no_grads_pool = autograd.grads.no_grads_pool.borrow_mut();
            let buf_any = no_grads_pool.cache.get(&buf.id()).unwrap();

            let buf1 = downcast_val::<f32, _, ()>(buf_any, &device).unwrap();
            assert_eq!(buf1.data.ptr, buf.data.ptr);
        }
    }

    #[test]
    fn test_buffer_creation_autograd_get_buf() {
        let device = CPU::<Autograd<Base>>::new();
        let buf: Buffer<f32, _> = Buffer::<f32, _>::new(&device, 10);

        let autograd = &device.modules;
        {
            let no_grads_pool = autograd.grads.no_grads_pool.borrow_mut();
            let buf1 = no_grads_pool
                .get_buf_with_dev::<f32, _, ()>(buf.id(), &device)
                .unwrap();
            assert_eq!(buf1.data.ptr, buf.data.ptr);
        }
    }

    #[test]
    fn test_buffer_creation_autograd_unregister() {
        let device = CPU::<Autograd<Base>>::new();
        let buf: Buffer<f32, _> = Buffer::<f32, _>::new(&device, 10);
        let id = buf.id();
        let autograd = &device.modules;

        drop(buf);

        {
            let no_grads_pool = autograd.grads.no_grads_pool.borrow_mut();
            assert!(no_grads_pool.cache.get(&id).is_none());
        }
    }

    #[test]
    fn test_buffer_new_and_retrieve() {
        let device = CPU::<Autograd<Base>>::new();
        let _lhs = Buffer::<f32, _>::new(&device, 10);
        
        for _ in 0..100 {
            let x = device.retrieve::<f32, ()>(100);
            assert_eq!(x.len(), 100)
        }

        let no_grads_pool = device.modules.grads.no_grads_pool.borrow();
        assert_eq!(no_grads_pool.cache.len(), 2);
    }

    
    #[test]
    fn test_cached_before_autograd() {
        // TODO: is a cached module is placed before Autograd results a problem
        // -> the retrieved buffer is not added to the no grads pool of the autograd module
        let device = CPU::<Cached<Autograd<Base>>>::new();
    }
}

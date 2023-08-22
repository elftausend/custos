mod gradients;
mod tape;

pub use gradients::*;
pub use tape::*;

use core::{
    any::Any,
    cell::{Ref, RefCell, RefMut},
    hash::BuildHasher,
    marker::PhantomData,
    mem::transmute,
};
use std::collections::HashMap;

use crate::{
    flag::AllocFlag, prelude::One, Alloc, Buffer, Device, HasId, Id, Module, OnDropBuffer,
    OnNewBuffer, Parents, PtrConv, Retrieve, Setup, Shape, TapeActions, UniqueId, WriteBuf, UnifiedMemChain, CachedCPU,
};

use super::{Cached, CachedModule};

#[derive(Debug, Default)]
pub struct Autograd<Mods> {
    _p: PhantomData<Mods>,
}

impl<Mods: Module<D>, D: Device + Default> Module<D> for Autograd<Mods> {
    type Module = AutogradModule<CachedModule<Mods::Module, D>, D>;

    #[inline]
    fn new() -> Self::Module {
        AutogradModule {
            modules: Cached::<Mods>::new(),
            tape: Default::default(),
        }
    }
}

impl<Mods> OnDropBuffer for Autograd<Mods> {}

pub struct AutogradModule<Mods, D> {
    pub modules: Mods,
    tape: RefCell<Tape<D>>,
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

impl<Mods, SD> AutogradModule<Mods, SD> {
    #[inline]
    pub fn register_no_grad_buf<T, D, S>(&self, buf: &Buffer<T, D, S>)
    where
        T: 'static,
        D: Device + PtrConv + 'static,
        S: Shape,
    {
        let no_grads_pool = &mut self.tape.borrow_mut().grads.no_grads_pool.cache;

        if no_grads_pool.get(&buf.id()).is_some() {
            return;
        }

        unsafe { register_buf(no_grads_pool, buf) };
    }
}

impl<T, D, S, Mods, SD> OnNewBuffer<T, D, S> for AutogradModule<Mods, SD>
where
    T: 'static,
    D: Alloc<T> + PtrConv + 'static,
    S: Shape,
    Mods: OnNewBuffer<T, D, S>,
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        self.register_no_grad_buf(new_buf);

        // allocates gradient memory for the corresponding buffer id
        // this prevents allocating with a non matching datatype
        // -> although retrieving the grads should fail if type information does not match
        // TODO: better solution?
        self.tape
            .borrow_mut()
            .grads
            .grads_pool
            .add_buf_once::<T, D, S>(device, new_buf.id());

        // pass down
        self.modules.on_new_buffer(device, new_buf)
    }
}

impl<Mods: OnDropBuffer, SD: Device> OnDropBuffer for AutogradModule<Mods, SD> {
    #[inline]
    fn on_drop_buffer<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        unregister_buf(
            &mut self.tape.borrow_mut().grads.no_grads_pool.cache,
            buf.id(),
        );
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<Mods: Setup<NewDev>, NewDev, D> Setup<NewDev> for AutogradModule<Mods, D> {
    #[inline]
    fn setup(device: &mut NewDev) {
        Mods::setup(device)
    }
}

impl<T: 'static, Mods: Retrieve<D, T>, D, SD: Device> Retrieve<D, T> for AutogradModule<Mods, SD>
where
    D: PtrConv + 'static,
{
    #[inline]
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> <D>::Data<T, S>
    where
        D: Alloc<T>,
        S: crate::Shape,
    {
        self.modules.retrieve(device, len, parents)
    }

    #[inline]
    fn on_retrieve_finish<S: Shape>(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        self.register_no_grad_buf(retrieved_buf);

        // allocates gradients
        self.tape
            .borrow_mut()
            .grads
            .grads_pool
            .add_buf_once::<T, D, S>(retrieved_buf.device(), retrieved_buf.id());

        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

impl<Mods, D> TapeActions<D> for AutogradModule<Mods, D> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<Tape<D>>> {
        Some(self.tape.borrow())
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<Tape<D>>> {
        Some(self.tape.borrow_mut())
    }
}

impl<Mods: UnifiedMemChain<D>, D: Device, SD> UnifiedMemChain<D> for AutogradModule<Mods, SD> {
    fn construct_unified_buf_from_cpu_buf<'a, T, S: Shape>(
        &self,
        device: &'a D,
        no_drop_buf: Buffer<'a, T, CachedCPU, S>
    ) -> crate::Result<Buffer<'a, T, D, S>>
    {
        self.modules.construct_unified_buf_from_cpu_buf(device, no_drop_buf)
    }
}

const AUTOGRAD_NOT_AVAILABLE: &'static str = "Autograd<> is not available.";

impl<'a, T, D, S> Buffer<'a, T, D, S>
where
    T: Clone + One + 'static,
    D: TapeActions<D> + WriteBuf<T, S, D> + Alloc<T> + 'static,
    S: Shape,
{
    /// Calls `.backward_seeded` on the [`Tape`].
    /// The seed of the gradient is set to `1` and contains `self.len()` elements.
    #[inline]
    pub fn backward(&self) {
        if let Some(mut tape) = self.device().tape_mut() {
            tape.backward_seeded(self)
        }
    }

    /// Returns a reference to the gradient of this buffer.
    /// The lifetime is bound to the lifetime of self, which is more strict and catches some mistakes at compile-time.
    /// However, If the borrow checker complains and you are sure that everything should be fine, use `grad_unbound` instead.
    ///
    /// Panics if the gradient was not allocated.
    #[inline]
    pub fn grad(&self) -> Ref<Self> {
        self.grad_unbound()
    }

    /// Returns a reference to the gradient of this buffer.
    /// Lifetimes are checked during runtime with `RefCell`.
    /// Panics if the gradient was not allocated.
    // TODO: Maybe return Result with two error variants?
    #[inline]
    pub fn grad_unbound(&self) -> Ref<'a, Self> {
        Ref::map(
            self.device().tape().expect(AUTOGRAD_NOT_AVAILABLE),
            |tape| {
                tape.grads.may_get_ref(self.id()).expect(
                "Gradient was not allocated for this buffer. Did you forget to call `backward`?",
            )
            },
        )
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// The lifetime is bound to the lifetime of self, which is more strict.
    /// If the borrow checker complains, use `grad_mut_unbound` instead.
    /// Panics if the gradient was not allocated.
    // TODO: Maybe return Result with two error variants?
    #[inline]
    pub fn grad_mut(&mut self) -> RefMut<Self> {
        self.grad_mut_unbound()
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// Lifetimes are checked during runtime.
    /// Panics if the gradient was not allocated.
    // TODO: Maybe return Result with two error variants?
    #[inline]
    pub fn grad_mut_unbound(&mut self) -> RefMut<'a, Self> {
        RefMut::map(
            self.device().tape_mut().expect(AUTOGRAD_NOT_AVAILABLE),
            |tape| {
                tape.grads.may_get_mut(self.id()).expect(
                "Gradient was not allocated for this buffer. Did you forget to call `backward`?",
            )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use core::any::Any;

    use crate::{Base, Buffer, Cached, Device, HasId, Retriever, Shape, CPU, TapeActions};

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
            let no_grads_pool = &mut autograd.tape.borrow_mut().grads.no_grads_pool;
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
            let no_grads_pool = &mut autograd.tape.borrow_mut().grads.no_grads_pool;
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
            let no_grads_pool = &autograd.tape.borrow_mut().grads.no_grads_pool;
            assert!(no_grads_pool.cache.get(&id).is_none());
        }
    }

    #[test]
    fn test_buffer_new_and_retrieve() {
        let device = CPU::<Autograd<Base>>::new();
        let _lhs = Buffer::<f32, _>::new(&device, 10);

        for _ in 0..100 {
            let x: Buffer<f32, _> = device.retrieve::<(), 0>(100, ());
            assert_eq!(x.len(), 100)
        }

        let no_grads_pool = &device.modules.tape.borrow().grads.no_grads_pool;
        assert_eq!(no_grads_pool.cache.len(), 2);
    }

    #[test]
    fn test_cached_before_autograd() {
        // is a cached module is placed before Autograd results a problem
        // -> the retrieved buffer is not added to the no grads pool of the autograd module
        let device = CPU::<Cached<Autograd<Base>>>::new();

        // how to fix this:
        // add retrieved buffer to no grads pool at the end of the chain (at device level (Retriever trait))
        // => "generator", "actor"

        let _lhs = Buffer::<f32, _>::new(&device, 10);

        for _ in 0..100 {
            let x: Buffer<f32, _> = device.retrieve::<(), 0>(100, ());
            assert_eq!(x.len(), 100)
        }

        let no_grads_pool = &device.modules.modules.tape.borrow().grads.no_grads_pool;
        assert_eq!(no_grads_pool.cache.len(), 2);
    }

    #[test]
    #[should_panic]
    fn test_tape_return_without_autograd() {
        let device = CPU::<Base>::new();
        let buf = Buffer::<f32, _>::new(&device, 10);
        buf.grad();
    }

    #[test]
    //#[should_panic]
    fn test_tape_return_without_grad_allocation() {
        let device = CPU::<Autograd<Base>>::new();
        let buf = Buffer::<f32, _>::new(&device, 10);

        // this does not panic anymore because grads are allocated if a new buffer is created (when using the Autograd module)
        device.add_grad_fn(|x, y| {
            
        });

        // device.tape().unwrap().backward(device)

        // buf.grad();
    }

    #[test]
    fn test_tape_return_with_grad_allocation() {
        let device = CPU::<Autograd<Base>>::new();
        let buf = Buffer::<f32, _>::new(&device, 10);

        // allocates a new gradient buffer if none exists for the specified id
        device
            .modules
            .tape
            .borrow_mut()
            .grads
            .get_mut::<f32, (), _>(&device, buf.id());

        // buf.grad();
    }
}

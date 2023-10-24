mod gradients;
mod tape;

pub use gradients::*;
pub use tape::*;

use core::cell::{Ref, RefCell, RefMut};

use crate::{
    pass_down_add_operation, pass_down_exec_now_module, prelude::One, register_buf, unregister_buf,
    Alloc, Buffer, Device, HasId, Module, OnDropBuffer, OnNewBuffer, Parents, PtrConv, Retrieve,
    RunModule, Setup, Shape, TapeActions, WriteBuf,
};

use super::{Cached, CachedModule};

#[derive(Debug, Default)]
pub struct Autograd<Mods> {
    pub modules: Mods,
    tape: RefCell<Tape>,
}

impl<Mods: Module<D>, D: Device> Module<D> for Autograd<Mods> {
    type Module = Autograd<CachedModule<Mods::Module, D>>;

    #[inline]
    fn new() -> Self::Module {
        Autograd {
            modules: Cached::<Mods>::new(),
            tape: Default::default(),
        }
    }
}

impl<Mods> Autograd<Mods> {
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

impl<T, D, S, Mods> OnNewBuffer<T, D, S> for Autograd<Mods>
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

impl<Mods: OnDropBuffer> OnDropBuffer for Autograd<Mods> {
    #[inline]
    fn on_drop_buffer<T, D: Device, S: Shape>(&self, device: &D, buf: &Buffer<T, D, S>) {
        unregister_buf(
            &mut self.tape.borrow_mut().grads.no_grads_pool.cache,
            buf.id(),
        );
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<Mods: Setup<NewDev>, NewDev> Setup<NewDev> for Autograd<Mods> {
    #[inline]
    fn setup(device: &mut NewDev) -> crate::Result<()> {
        Mods::setup(device)
    }
}

impl<T: 'static, Mods: Retrieve<D, T>, D> Retrieve<D, T> for Autograd<Mods>
where
    D: PtrConv + Device + 'static,
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

impl<Mods> TapeActions for Autograd<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<Tape>> {
        Some(self.tape.borrow())
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<Tape>> {
        Some(self.tape.borrow_mut())
    }
}

impl<Mods: RunModule<D>, D> RunModule<D> for Autograd<Mods> {
    #[inline]
    fn run(&self, _device: &D) -> crate::Result<()> {
        self.modules.run(_device)
    }
}

pass_down_add_operation!(Autograd);
pass_down_exec_now_module!(Autograd);

const AUTOGRAD_NOT_AVAILABLE: &str = "Autograd<> is not available.";

impl<'a, T, D, S> Buffer<'a, T, D, S>
where
    T: Clone + One + 'static,
    D: TapeActions + WriteBuf<T, S, D> + Alloc<T> + 'static,
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
    pub fn grad_mut(&self) -> RefMut<Self> {
        self.grad_mut_unbound()
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// Lifetimes are checked during runtime.
    /// Panics if the gradient was not allocated.
    // TODO: Maybe return Result with two error variants?
    #[inline]
    pub fn grad_mut_unbound(&self) -> RefMut<'a, Self> {
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

    use crate::{
        Base, Buffer, Cached, Device, HasId, MayTapeActions, Module, Retriever, Shape, TapeActions,
        CPU,
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
    #[cfg_attr(miri, ignore)]
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
    #[cfg_attr(miri, ignore)]
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
        let device: CPU<Autograd<crate::CachedModule<Base, CPU>>> = CPU::<Autograd<Base>>::new();
        let buf = Buffer::<f32, _>::new(&device, 10);

        let out = Buffer::<f32, _>::new(&device, 10);

        let ids = (buf.id(), out.id());
        // this does not panic anymore because grads are allocated if a new buffer is created (when using the Autograd module)
        device.add_grad_fn(move |grads| {
            let (_buf, buf_grad, _out) =
                grads.get_double::<f32, (), (), CPU<Autograd<crate::CachedModule<Base, CPU>>>>(ids);
            for val in buf_grad.as_mut_slice() {
                *val = 5.;
            }
        });

        device.tape_mut().unwrap().backward_seeded(&out);

        assert_eq!(&***buf.grad(), [5.; 10]);
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

        buf.grad();
    }

    #[cfg(feature = "autograd")]
    pub trait UnaryByMods<Mods> {
        fn unary_ew(&self, mods: &Mods);
    }

    #[cfg(feature = "autograd")]
    impl<Mods: MayTapeActions + 'static> UnaryByMods<Mods> for CPU {
        fn unary_ew(&self, mods: &Mods) {
            mods.add_grad_fn(|_grads| {});
        }
    }

    #[cfg(feature = "autograd")]
    #[test]
    fn test_autograd_by_separate_module() {
        let autograd = <Autograd<Base> as Module<CPU>>::new();
        let device = CPU::<Base>::new();
        device.unary_ew(&autograd);
    }
}

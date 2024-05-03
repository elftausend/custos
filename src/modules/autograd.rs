mod gradients;
mod tape;
mod wrapper;

pub use gradients::*;
pub use tape::*;

use core::cell::{Cell, UnsafeCell};

use crate::{
    impl_remove_layer, pass_down_add_operation, pass_down_cached_buffers, pass_down_cursor,
    pass_down_exec_now_module, pass_down_replace_buf_module, register_buf_copyable,
    unregister_buf_copyable, AddGradFn, AddLayer, Alloc, Buffer, Device, HasId, IsShapeIndep,
    Module, OnDropBuffer, OnNewBuffer, Parents, Retrieve, RunModule, Setup, ShallowCopy, Shape,
    TapeActions,
};

use self::wrapper::ReqGradWrapper;

#[derive(Debug, Default)]
pub struct Autograd<Mods> {
    pub modules: Mods,
    /// Caches gradients for each [`Buffer`]'s id ([`Ident`]).
    pub grads: UnsafeCell<Gradients>, // could use RefCell
    tape: UnsafeCell<Tape>,
    pub enabled: Cell<bool>,
}

impl<Mods: Module<D>, D: Device> Module<D> for Autograd<Mods> {
    // type Module = Autograd<CachedModule<Mods::Module, D>>;
    type Module = Autograd<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        Autograd {
            // modules: Cached::<Mods>::new(),
            modules: Mods::new(),
            grads: Default::default(),
            tape: Default::default(),
            enabled: Cell::new(true),
        }
    }
}

impl<Mods> Autograd<Mods> {
    #[inline]
    pub fn register_no_grad_buf<T, D, S>(&self, buf: &Buffer<T, D, S>)
    where
        T: 'static,
        D: Device + IsShapeIndep + 'static,
        D::Data<T, S>: ShallowCopy,
        S: Shape,
    {
        let no_grads_pool = unsafe { &mut (*self.grads.get()).no_grads_pool };

        if no_grads_pool.get(&buf.id()).is_some() {
            return;
        }

        unsafe { register_buf_copyable(no_grads_pool, buf) };
    }
}

impl<T, D, Mods, S: Shape> OnNewBuffer<T, D, S> for Autograd<Mods>
where
    T: 'static,
    D: Alloc<T> + IsShapeIndep + 'static,
    D::Data<T, S>: ShallowCopy,
    Mods: OnNewBuffer<T, D, S>,
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        unsafe {
            (*self.grads.get())
                .buf_requires_grad
                .insert(*new_buf.id(), new_buf.requires_grad())
        };
        self.register_no_grad_buf(new_buf);

        // pass down
        self.modules.on_new_buffer(device, new_buf)
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Autograd<Mods> {
    #[inline]
    fn on_drop_buffer<T, D: Device, S: Shape>(&self, device: &D, buf: &Buffer<T, D, S>) {
        unsafe { (*self.grads.get()).buf_requires_grad.remove(&*buf.id()) };
        unregister_buf_copyable(unsafe { &mut (*self.grads.get()).no_grads_pool }, buf.id());

        // TODO
        // FIXME if an alloc flag None buffer goes out of scope and it has used it's gradient buffer before,
        // the gradient buffer will stay allocated
        // - deallocate directly -> however, a user storing the id maybe wants to retrieve the grad buf
        // - add to id set of potentially unused buffers

        self.modules.on_drop_buffer(device, buf)
    }
}

pub trait HasAutograd {}
impl<Mods> HasAutograd for Autograd<Mods> {}

impl<Mods: Setup<NewDev>, NewDev> Setup<NewDev> for Autograd<Mods> {
    #[inline]
    fn setup(device: &mut NewDev) -> crate::Result<()> {
        Mods::setup(device)
    }
}

impl_remove_layer!(Autograd);

impl<NewMods, SD> AddLayer<NewMods, SD> for Autograd<()> {
    type Wrapped = crate::Autograd<NewMods>;

    #[inline]
    fn wrap_layer(inner_mods: NewMods) -> Self::Wrapped {
        Autograd {
            modules: inner_mods,
            grads: Default::default(),
            tape: Default::default(),
            enabled: Cell::new(true),
        }
    }
}

impl<T: 'static, Mods: Retrieve<D, T, S>, D, S: Shape> Retrieve<D, T, S> for Autograd<Mods>
where
    D: IsShapeIndep + Device + 'static,
    D::Data<T, S>: ShallowCopy,
{
    #[inline]
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<T, D::Base<T, S>>>
    where
        D: Alloc<T>,
    {
        let requires_grad = parents.requires_grads().iter().any(|&x| x);
        let data = self.modules.retrieve(device, len, parents)?;
        unsafe {
            (*self.grads.get())
                .buf_requires_grad
                .insert(*data.id(), requires_grad)
        };

        Ok(ReqGradWrapper {
            requires_grad,
            data,
            _pd: core::marker::PhantomData,
        })
    }

    #[inline]
    fn on_retrieve_finish(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        self.register_no_grad_buf(retrieved_buf);

        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

pass_down_cursor!(Autograd);

impl<Mods> TapeActions for Autograd<Mods> {
    #[inline]
    unsafe fn tape(&self) -> Option<&Tape> {
        Some(&*self.tape.get())
        // Some(self.tape.borrow())
    }

    #[inline]
    unsafe fn tape_mut(&self) -> Option<&mut Tape> {
        Some(&mut *self.tape.get())
        // Some(unsafe {&mut (self.tape.get_mut()) })
        // Some(self.tape.borrow_mut())
    }

    unsafe fn gradients(&self) -> Option<&crate::Gradients> {
        Some(&*self.grads.get())
    }

    unsafe fn gradients_mut(&self) -> Option<&mut crate::Gradients> {
        Some(&mut *self.grads.get())
    }
}

impl<Mods: RunModule<D>, D> RunModule<D> for Autograd<Mods> {
    #[inline]
    fn run(&self, _device: &D) -> crate::Result<()> {
        self.modules.run(_device)
    }
}

impl<Mods: AddGradFn> AddGradFn for Autograd<Mods> {
    #[inline]
    fn add_grad_fn<Args: Parents<N> + crate::UpdateArgs, const N: usize>(
        &self,
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    ) {
        if !self.enabled.get() {
            return;
        }
        unsafe { (*self.tape.get()).add_grad_fn(args, op) }
    }
    #[inline]
    fn is_grad_enabled(&self) -> bool {
        self.enabled.get()
    }

    #[inline]
    fn set_grad_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
    }
}

pass_down_add_operation!(Autograd);
pass_down_exec_now_module!(Autograd);
pass_down_cached_buffers!(Autograd);
pass_down_replace_buf_module!(Autograd);

#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {
    use crate::{
        AddGradFn, Base, BoxedShallowCopy, Buffer, Cached, Combiner, Cursor, Device, HasId, Lazy,
        Module, Retriever, Shape, UnaryGrad, CPU,
    };

    use super::Autograd;

    #[inline]
    pub fn downcast_val<'a, 'b, T: 'static, D: Device + 'static, S: Shape>(
        buf_any: &'b Box<dyn BoxedShallowCopy>,
        _device: &'a D,
    ) -> Option<&'b Buffer<'a, T, D, S>> {
        buf_any.as_any().downcast_ref::<Buffer<T, D, S>>()
    }

    #[test]
    fn test_buffer_creation_autograd_register_manual() {
        let device = CPU::<Autograd<Cached<Base>>>::new();
        let buf: Buffer<f32, _> = Buffer::<f32, _>::new(&device, 10);

        let autograd = &device.modules;
        {
            let no_grads_pool = unsafe { &(*autograd.grads.get()).no_grads_pool };
            // let no_grads_pool = &mut autograd.tape.grads.no_grads_pool;
            let buf_any = no_grads_pool.get(&buf.id()).unwrap();

            let buf1 = downcast_val::<f32, _, ()>(buf_any, &device).unwrap();
            assert_eq!(buf1.base().ptr, buf.base().ptr);
        }
    }

    #[test]
    fn test_buffer_creation_autograd_get_buf() {
        let device: CPU<Autograd<crate::CachedModule<Base, CPU>>> =
            CPU::<Autograd<Cached<Base>>>::new();
        let buf: Buffer<f32, _> = Buffer::<f32, _>::new(&device, 10);

        let autograd = &device.modules;
        {
            let no_grads_pool = unsafe { &mut (*autograd.grads.get()).no_grads_pool };
            let buf1: &Buffer<f32, CPU<Autograd<crate::CachedModule<Base, CPU>>>> = no_grads_pool
                .get(&buf.id())
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();
            // let no_grads_pool = &mut autograd.tape.borrow_mut().grads.no_grads_pool;
            assert_eq!(buf1.base().ptr, buf.base().ptr);
        }
    }

    #[test]
    fn test_buffer_creation_autograd_unregister() {
        let device = CPU::<Autograd<Cached<Base>>>::new();
        let buf: Buffer<f32, _> = Buffer::<f32, _>::new(&device, 10);
        let id = buf.id();
        let autograd = &device.modules;

        drop(buf);

        {
            let no_grads_pool = unsafe { &(*autograd.grads.get()).no_grads_pool };
            // let no_grads_pool = &autograd.tape.borrow_mut().grads.no_grads_pool;
            assert!(no_grads_pool.get(&id).is_none());
        }
    }

    #[test]
    fn test_buffer_new_and_retrieve() {
        let device = CPU::<Autograd<Cached<Base>>>::new();
        let _lhs = Buffer::<f32, _>::new(&device, 10);

        for _ in device.range(0..100) {
            let x: Buffer<f32, _> = device.retrieve::<0>(100, ()).unwrap();
            assert_eq!(x.len(), 100)
        }

        let no_grads_pool = unsafe { &(*device.modules.grads.get()).no_grads_pool };
        // let no_grads_pool = &device.modules.tape.borrow().grads.no_grads_pool;
        assert_eq!(no_grads_pool.len(), 2);
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

        for _ in device.range(0..100) {
            let x: Buffer<f32, _> = device.retrieve::<0>(100, ()).unwrap();
            assert_eq!(x.len(), 100)
        }

        let no_grads_pool = unsafe { &(*device.modules.modules.grads.get()).no_grads_pool };
        // let no_grads_pool = &device.modules.modules.tape.borrow().grads.no_grads_pool;
        assert_eq!(no_grads_pool.len(), 2);
    }

    #[test]
    #[should_panic]
    fn test_tape_return_without_autograd() {
        let device = CPU::<Base>::new();
        let buf = Buffer::<f32, _>::new(&device, 10);
        buf.grad();
    }

    #[test]
    fn test_grad_fn_with_lazy_buffer_source_but_no_true_lazy() {
        let device = CPU::<Autograd<Lazy<Base>>>::new();
        let buf = Buffer::<f32, _>::new(&device, 10).require_grad();

        let out = Buffer::<f32, _>::new(&device, 10);

        device.add_grad_fn((&buf, &out), |(buf, _out)| {
            for (val, grad) in buf.grad_mut().iter_mut().zip(_out.grad().iter()) {
                *val = 5. * grad;
            }
            Ok(())
        });

        out.backward();

        assert_eq!(&***buf.grad(), [5.; 10]);
    }

    #[test]
    fn test_tape_return_with_grad_allocation() {
        let device = CPU::<Autograd<Base>>::new();
        let buf = Buffer::<f32, _>::new(&device, 10);

        // let no_grads_pool = unsafe { &(*device.modules.tape.get()).grads };
        // allocates a new gradient buffer if none exists for the specified id
        // device
        //     .modules
        //     .tape
        //     .borrow_mut()
        //     .grads
        //     .get_mut::<f32, (), _>(&device, buf.id());

        buf.grad();
    }

    #[cfg(feature = "autograd")]
    pub trait UnaryByMods<Mods> {
        fn unary_ew(&self, mods: &Mods);
    }

    #[cfg(feature = "autograd")]
    impl<Mods: AddGradFn + 'static> UnaryByMods<Mods> for CPU {
        fn unary_ew(&self, mods: &Mods) {
            mods.add_grad_fn((), |_| Ok(()));
        }
    }

    #[cfg(feature = "autograd")]
    #[test]
    fn test_autograd_by_separate_module() {
        let autograd = <Autograd<Base> as Module<CPU>>::new();
        let device = CPU::<Base>::new();
        device.unary_ew(&autograd);
    }

    #[test]
    fn test_grad_new_api() {
        use crate::AddGradFn;

        let device = CPU::<Autograd<Base>>::new();

        let lhs = device.buffer([1, 2, 3, 4]).require_grad();
        let out = lhs.empty_like();

        device.add_grad_fn((&lhs, &out), |(lhs, out)| {
            // lhs.grad();
            lhs.device()
                .add_unary_grad(lhs, lhs.grad_mut(), out.grad(), |x| x.add(3));
            // lhs.device().add_ew_grad(lhs.grad(), rhs.grad(), out.grad());
            Ok(())
        });

        out.backward();

        assert_eq!(lhs.try_grad().unwrap().as_slice(), [4, 5, 6, 7]);
    }

    #[test]
    fn test_autograd_disabling() {
        let device = CPU::<Autograd<Base>>::new();

        let lhs = device.buffer([1, 2, 3, 4]).require_grad();
        let out = lhs.empty_like();

        device.disable_grad();

        device.add_grad_fn((&lhs, &out), |(lhs, out)| {
            lhs.device()
                .add_unary_grad(lhs, lhs.grad_mut(), out.grad(), |x| x.add(3));
            panic!("should not be called");
        });

        out.backward();

        assert!(lhs.try_grad().is_none());

        device.enable_grad();

        device.add_grad_fn((&lhs, &out), |(lhs, out)| {
            lhs.device()
                .add_unary_grad(lhs, lhs.grad_mut(), out.grad(), |x| x.add(3));
            Ok(())
        });

        out.backward();

        assert_eq!(lhs.try_grad().unwrap().as_slice(), [4, 5, 6, 7]);
    }

    #[test]
    fn test_req_grad_chaining() {
        let device = CPU::<Autograd<Base>>::new();

        let lhs = device.buffer([1i32, 2, 3, 4]).require_grad();
        assert!(*unsafe {
            (*device.modules.grads.get())
                .buf_requires_grad
                .get(&*lhs.id())
                .unwrap()
        });
        assert!(lhs.requires_grad());

        let no_grad = device.buffer([1i32, 2, 3, 4]).no_grad();
        assert!(!*unsafe {
            (*device.modules.grads.get())
                .buf_requires_grad
                .get(&*no_grad.id())
                .unwrap()
        });
        let rhs = device.buffer([1i32, 2, 3, 4]).no_grad();
        assert!(!*unsafe {
            (*device.modules.grads.get())
                .buf_requires_grad
                .get(&*rhs.id())
                .unwrap()
        });
        assert!(!rhs.requires_grad());

        let out: Buffer<i32, _> = device.retrieve(rhs.len(), (&lhs, &rhs)).unwrap();
        assert!(*unsafe {
            (*device.modules.grads.get())
                .buf_requires_grad
                .get(&*out.id())
                .unwrap()
        });
        assert!(out.requires_grad());

        let out: Buffer<i32, _> = device.retrieve(rhs.len(), &lhs).unwrap();
        assert!(out.requires_grad());

        let out: Buffer<i32, _> = device.retrieve(rhs.len(), &rhs).unwrap();
        assert!(!out.requires_grad());

        let out: Buffer<i32, _> = device.retrieve(rhs.len(), (&no_grad, &rhs)).unwrap();
        assert!(!out.requires_grad());
    }
}

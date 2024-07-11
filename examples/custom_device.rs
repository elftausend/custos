use core::cell::RefCell;
use core::fmt;
use std::{
    any::{Any, TypeId},
    cell::{Cell, UnsafeCell},
    collections::HashMap,
    hash::BuildHasherDefault,
    marker::PhantomData,
    mem::transmute,
    ops::{AddAssign, Mul},
};

use custos::{
    flag::AllocFlag, AddOperation, Alloc, Base, Buffer, CachingError, Device, HasId, Id, NoHasher,
    OnDropBuffer, OnNewBuffer, PtrType, Retrieve, Retriever, Shape, UniqueId, Unit, WrappedData,
    CPU,
};

pub trait Module<'a, D: 'a, Mods = ()> {
    type Module;

    fn new() -> Self::Module;
}

pub trait Str {
    fn str(&self) -> &String;
}

pub trait New<SimpleMods> {
    fn new1<'a, NewMods>() -> CPU<SimpleMods::Module>
    where
        Self: 'a,
        SimpleMods: Module<'a, CPU, Module = NewMods>;
}

impl<SimpleMods> New<SimpleMods> for CPU<SimpleMods> {
    #[inline]
    fn new1<'a, NewMods>() -> CPU<SimpleMods::Module>
    where
        Self: 'a,
        SimpleMods: Module<'a, CPU, Module = NewMods>,
    {
        CPU {
            modules: SimpleMods::new(),
        }
    }
}

#[derive(Default)]
pub struct Autograd<'a, Mods> {
    _cache: UnsafeCell<BorrowCache<'a>>,
    val: Cell<Option<&'a f32>>,
    _modules: Mods,
}

impl<'a, T, S: Shape, D: Device, Mods: OnDropBuffer> Retrieve<D, T, S> for Autograd<'a, Mods> {
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl custos::Parents<NUM_PARENTS>,
    ) -> custos::Result<Self::Wrap<T, <D>::Base<T, S>>>
    where
        S: Shape,
        D: Device + Alloc<T>,
    {
        todo!()
    }
}

impl<'a, Mods: OnDropBuffer> AddOperation for Autograd<'a, Mods> {
    fn add_op<Args: custos::Parents<N> + custos::UpdateArgs, const N: usize>(
        &self,
        args: Args,
        operation: fn(&mut Args) -> custos::Result<()>,
    ) -> custos::Result<()> {
        todo!()
    }

    fn ops_count(&self) -> usize {
        todo!()
    }

    fn set_lazy_enabled(&self, enabled: bool) {
        todo!()
    }

    fn is_lazy_enabled(&self) -> bool {
        todo!()
    }
}

impl<'a, Mods> Autograd<'a, Mods> {
    pub fn add_buf<OtherMods>(&'a self, device: &'a CPU<OtherMods>) {
        // unsafe { (*self._cache.get()).add_buf(device) };
        // binding.get_buf_mut(device);
        // self.val.set(Some(&device.val));
    }
}
pub trait GradActions<'dev, D: Device> {
    fn get_grad<T, S>(&self, for_buf_id: Id) -> &Buffer<'dev, T, D, S>
    where
        T: 'static,
        S: Shape;

    #[allow(clippy::mut_from_ref)]
    unsafe fn get_grad_mut<'b, T, S>(&'b self, for_buf_id: Id) -> &'b mut Buffer<'dev, T, D, S>
    where
        T: 'static,
        S: Shape;

    fn grad<T, S>(&self, for_buf: &Buffer<'_, T, D, S>) -> &Buffer<'dev, T, D, S>
    where
        T: 'static,
        S: Shape,
    {
        self.get_grad(for_buf.id())
    }

    fn grad_mut<'b, T, S>(
        &'b self,
        for_buf: &'b Buffer<'_, T, D, S>,
    ) -> &'b mut Buffer<'dev, T, D, S>
    where
        T: 'static,
        S: Shape,
    {
        todo!()
        // self.get_grad_mut(for_buf.id())
    }
}

impl<'dev, Mods, D: Device + 'static> GradActions<'dev, D> for Autograd<'dev, Mods> {
    fn get_grad<'a, T, S>(&'a self, for_buf_id: Id) -> &'a Buffer<'dev, T, D, S>
    where
        T: 'static,
        S: Shape,
    {
        unsafe { (*self._cache.get()).get_buf(for_buf_id) }.unwrap()
    }

    unsafe fn get_grad_mut<'b, T, S>(&'b self, for_buf_id: Id) -> &'b mut Buffer<'dev, T, D, S>
    where
        T: 'static,
        S: Shape,
    {
        unsafe { (*self._cache.get()).get_buf_mut(for_buf_id) }.unwrap()
    }
}

impl<'dev, Mods: OnDropBuffer + GradActions<'dev, Self>> GradActions<'dev, Self> for CPU<Mods>
where
    Self: 'dev,
{
    fn get_grad<'a, T, S>(&'a self, for_buf_id: Id) -> &'a Buffer<'dev, T, Self, S>
    where
        T: 'static,
        S: Shape,
    {
        self.modules.get_grad(for_buf_id)
    }

    unsafe fn get_grad_mut<'b, T, S>(&'b self, for_buf_id: Id) -> &'b mut Buffer<'dev, T, Self, S>
    where
        T: 'static,
        S: Shape,
    {
        self.modules.get_grad_mut(for_buf_id)
    }
}

pub trait AnyBuffer {
    fn type_id(&self) -> TypeId;
}

impl<'a, T, D, S> AnyBuffer for Buffer<'a, T, D, S>
where
    T: 'static,
    D: Device + 'static,
    S: Shape,
{
    #[inline]
    fn type_id(&self) -> TypeId {
        TypeId::of::<Buffer<T, D, S>>()
    }
}

impl fmt::Debug for dyn AnyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Any").finish_non_exhaustive()
    }
}
impl<'a> dyn AnyBuffer + 'a {
    pub fn is<T: 'static>(&self) -> bool {
        std::any::TypeId::of::<T>() == self.type_id()
    }

    #[inline]
    pub unsafe fn downcast_mut_unchecked<T>(&mut self) -> &mut T {
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &mut *(self as *mut (dyn AnyBuffer + 'a) as *mut T) }
    }

    #[inline]
    pub unsafe fn downcast_ref_unchecked<T: 'static>(&self) -> &T {
        debug_assert!(self.is::<T>());
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &*(self as *const (dyn AnyBuffer + 'a) as *const T) }
    }

    #[inline]
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            Some(unsafe { self.downcast_mut_unchecked() })
        } else {
            None
        }
    }

    #[inline]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        if self.is::<T>() {
            Some(unsafe { self.downcast_ref_unchecked() })
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct BorrowCache<'a> {
    cache: HashMap<UniqueId, Box<(dyn AnyBuffer + 'a)>, BuildHasherDefault<NoHasher>>,
}

impl<'dev> BorrowCache<'dev> {
    pub fn add_buf_once<T, D, S>(&mut self, device: &'dev D, id: Id, new_buf: &mut bool)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        if self.cache.contains_key(&id) {
            return;
        }
        *new_buf = true;
        self.add_buf::<T, D, S>(device, id)
    }

    pub fn add_buf<T, D, S>(&mut self, device: &'dev D, id: Id)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        let buf = Buffer {
            data: device.base_to_data(device.alloc::<S>(id.len, AllocFlag::BorrowedCache).unwrap()),
            device: Some(device),
        };
        self.cache.insert(*id, Box::new(buf));
    }

    #[inline]
    pub fn get_buf<'a, T, D, S>(&'a self, id: Id) -> Result<&'a Buffer<'dev, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        self.cache
            .get(&id)
            .ok_or(CachingError::InvalidId)?
            .downcast_ref()
            .ok_or(CachingError::InvalidTypeInfo)
    }

    #[inline]
    pub fn get_buf_mut<'a, T, D, S>(
        &'a mut self,
        id: Id,
    ) -> Result<&'a mut Buffer<'dev, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        let dyn_buf = self.cache.get_mut(&id).ok_or(CachingError::InvalidId)?;

        if !dyn_buf.is::<Buffer<T, D, S>>() {
            return Err(CachingError::InvalidTypeInfo);
        }
        Ok(unsafe { dyn_buf.downcast_mut_unchecked() })
    }
}

pub struct Test<'a> {
    pd: PhantomData<&'a ()>,
}

impl<'a, D: 'a, Mods: Module<'a, D>> Module<'a, D> for Autograd<'a, Mods> {
    type Module = Autograd<'a, Mods::Module>;

    fn new() -> Self::Module {
        Autograd {
            _cache: Default::default(),
            _modules: Mods::new(),
            val: Default::default(),
        }
    }
}

impl<'a, T, D, S, Mods: OnNewBuffer<T, D, S>> OnNewBuffer<T, D, S> for Autograd<'a, Mods>
where
    D: Device,
    S: Shape,
{
}

impl<'a, Mods: OnDropBuffer> OnDropBuffer for Autograd<'a, Mods> {
    #[inline]
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(
        &self,
        _device: &D,
        _buf: &custos::prelude::Buffer<T, D, S>,
    ) {
        self._modules.on_drop_buffer(_device, _buf)
    }
}

impl<'a, Mods: WrappedData> WrappedData for Autograd<'a, Mods> {
    type Wrap<T, Base: HasId + PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        self._modules.wrap_in_base(base)
    }

    #[inline]
    fn wrapped_as_base<T, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: HasId + PtrType>(wrap: &mut Self::Wrap<T, Base>) -> &mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<'a, D: 'a> Module<'a, D> for Base {
    type Module = Base;

    fn new() -> Self::Module {
        Base
    }
}

pub trait Grad<'dev, T, D: Device, S: Shape> {
    fn grad1(&self) -> &Buffer<'dev, T, D, S>;
    fn grad_mut1(&mut self) -> &mut Buffer<'dev, T, D, S>;
}

impl<'dev, T, D, S> Grad<'dev, T, D, S> for Buffer<'_, T, D, S>
where
    T: 'static,
    D: Device + 'static + GradActions<'dev, D>,
    S: Shape,
{
    fn grad1(&self) -> &Buffer<'dev, T, D, S> {
        self.device().get_grad(self.id())
    }

    fn grad_mut1(&mut self) -> &mut Buffer<'dev, T, D, S> {
        unsafe { self.device().get_grad_mut(self.id()) }
    }
}

pub trait AddBuf<'dev, T: Unit, S: Shape = (), D: Device = Self>: Sized + Device {
    fn add(&self, lhs: &mut Buffer<T, D, S>, rhs: &mut Buffer<T, D, S>) -> Buffer<T, Self, S>;
    fn test<'a>(&self, lhs: &'a Buffer<'_, T, D, S>) -> &'a Buffer<'dev, T, Self, S>;
}

impl<'dev, T, S, Mods> AddBuf<'dev, T, S, Self> for CPU<Mods>
where
    T: Unit + Copy + AddAssign + 'static,
    S: Shape,
    Mods: 'static + GradActions<'dev, Self> + OnDropBuffer + AddOperation + Retrieve<Self, T, S>,
{
    fn add(
        &self,
        lhs: &mut Buffer<T, Self, S>,
        rhs: &mut Buffer<T, Self, S>,
    ) -> Buffer<T, Self, S> {
        let out = self.retrieve(lhs.len, (&*lhs, &*rhs)).unwrap();

        // lazy fn not grad fn -> wurscht
        self.add_op((lhs, rhs, &out), |(lhs, rhs, out)| {
            add_ew_grad_slice(lhs.grad_mut1(), out.grad1());
            add_ew_grad_slice(rhs.grad_mut1(), out.grad1());
            Ok(())
        })
        .unwrap();

        out
    }

    fn test<'a>(&self, lhs: &'a Buffer<'_, T, Self, S>) -> &'a Buffer<'dev, T, Self, S> {
        lhs.grad1()
    }
}

fn add_ew_grad_slice<T: Copy + AddAssign>(grad_acc: &mut [T], out_grad: &[T]) {
    for (grad, out_grad) in grad_acc.iter_mut().zip(out_grad) {
        *grad += *out_grad;
    }
}

#[derive(Default)]
pub struct Typ {
    x: i32,
}

fn main() {
    // let x = Box::new(Typ::default());
    // Box::into_raw(x);
    //
    {
        let dev = CPU::<Autograd<Base>>::new1();

        let mut out = dev.buffer([1, 2, 3]);
        let mut out1 = dev.buffer([1, 2, 3]);

        let mut out = dev.add(&mut out, &mut out1);
        dev.add(&mut out, &mut out1);
        dev.test(&out);

        // dev.get_grad::<i32, ()>(out.id());
        {
            let z = out.grad_mut1();
            let x = out1.grad_mut1();
            assert_eq!(z.len(), x.len());
            out.grad1();
        }

        let x = dev.grad_mut(&out);
        let z = dev.grad_mut(&out);
        assert_eq!(z.len(), x.len());
        unsafe { dev.get_grad_mut::<i32, ()>(out.id()) };
        unsafe { dev.get_grad_mut::<i32, ()>(out.id()) };
    }

    // return;

    // let out = Buffer::new(&dev, 10);
    //
    // out.grad();

    let mods = Autograd::<Base>::default();
    {
        let dev = CPU::<Autograd<Base>>::new1();
        let mut cache = BorrowCache::default();
        cache.add_buf::<i32, _, ()>(&dev, Id { id: 0, len: 10 });
        // dev.modules.add_buf(&dev);
        // let out = dev.modules._cache._cache.get(&3).unwrap();
        // mods.add_buunsafe { f(&dev);
        // mods.add_buf(&dev);
        {
            // cache.add_buf(&dev);
        }
        {
            // cache.add_buf(&dev);
        }
        // cache.get_buf_mut(&dev);
        let out = cache
            .get_buf::<i32, CPU<Autograd<Base>>, ()>(Id { id: 0, len: 10 })
            .unwrap();
        let out1 = cache
            .get_buf_mut::<i32, CPU<Autograd<Base>>, ()>(Id { id: 0, len: 10 })
            .unwrap();
        // assert_eq!(out.len(), out1.len());
        out1;
    }
    // let out = unsafe { cache.get_buf::<i33, CPU<Autograd<Base>>, ()>(Id { id: 0, len: 10 }) };
    let dev = CPU::<Autograd<Base>>::new1();
    // cache.add_buf(&dev);
    // mods.val;
}

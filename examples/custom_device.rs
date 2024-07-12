use std::{
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    ops::AddAssign,
};

use custos::{
    AddOperation, Alloc, Autograd, Base, BorrowCacheLT, Buffer, Cached, Device, HasId, Id, Module, OnDropBuffer, OnNewBuffer, PtrType, Retrieve, Retriever, Shape, Unit, WrappedData, CPU
};

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
    _cache: UnsafeCell<BorrowCacheLT<'a>>,
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

impl<'a, 'b, T, D, S, Mods: OnNewBuffer<'a, T, D, S>> OnNewBuffer<'b, T, D, S> for Autograd<'a, Mods>
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

fn main() {
    // let x = Box::new(Typ::default());
    // Box::into_raw(x);
    //
    {
        let dev = CPU::<Autograd<Cached<Base>>>::new1();

        // Buffer::<f32, _>::new(&dev, 10);
        let data = dev.wrap_in_base(dev.alloc::<()>(10, custos::flag::AllocFlag::None).unwrap());
        let buffer: Buffer<f32, _> = Buffer {
            data,
            device: Some(&dev),
        };
        dev.on_new_buffer(&dev, &buffer);

        // let mut out = dev.buffer([1, 2, 3]);
        // let mut out1 = dev.buffer([1, 2, 3]);

        // let mut out = dev.add(&mut out, &mut out1);
        // dev.add(&mut out, &mut out1);
        // dev.test(&out);

        // // dev.get_grad::<i32, ()>(out.id());
        // {
        //     let z = out.grad_mut1();
        //     let x = out1.grad_mut1();
        //     assert_eq!(z.len(), x.len());
        //     out.grad1();
        // }

        // let x = dev.grad_mut(&out);
        // let z = dev.grad_mut(&out);
        // assert_eq!(z.len(), x.len());
        // unsafe { dev.get_grad_mut::<i32, ()>(out.id()) };
        // unsafe { dev.get_grad_mut::<i32, ()>(out.id()) };
    }

    // return;

    // let out = Buffer::new(&dev, 10);
    //
    // out.grad();

    let mods = Autograd::<Base>::default();
    {
        let dev = CPU::<Autograd<Base>>::new1();
        let mut cache = BorrowCacheLT::default();
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

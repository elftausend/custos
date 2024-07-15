use std::{
    cell::{Cell, RefCell, UnsafeCell},
    collections::HashMap,
    convert::Infallible,
    marker::PhantomData,
    ops::AddAssign,
};

use custos::{
    cpu::CPUPtr, flag::AllocFlag, impl_device_traits, AddGradFn, AddOperation, Alloc, Base,
    BorrowCacheLT, Buffer, Cached, CachedModule, Device, DeviceError, DevicelessAble, HasId, Id,
    LazyGraph2, Module, OnDropBuffer, OnNewBuffer, PtrType, Retrieve, Retriever, Setup, Shape,
    TapeActions, Tape, Unit, WrappedData, CPU,
};

pub trait Str {
    fn str(&self) -> &String;
}

#[derive(Default)]
pub struct CPU2<'a, Mods: 'a = Base> {
    pub modules: Mods,
    pub graph: LazyGraph2<'a>,
    pd: PhantomData<&'a ()>,
}

impl<'dev, T, D, S, Mods> crate::OnNewBuffer<'dev, T, D, S> for CPU2<'_, Mods>
where
    Self: 'dev,
    T: crate::Unit,
    D: Device,
    S: Shape,
    Mods: crate::OnNewBuffer<'dev, T, D, S>,
{
    #[inline]
    fn on_new_buffer(&self, device: &'dev D, new_buf: &Buffer<'dev, T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}

impl<'dev, Mods: crate::OnDropBuffer> crate::OnDropBuffer for CPU2<'dev, Mods> {
    #[inline]
    fn on_drop_buffer<T: crate::Unit, D: Device, S: Shape>(
        &self,
        device: &D,
        buf: &Buffer<T, D, S>,
    ) {
        self.modules.on_drop_buffer(device, buf)
    }
}
impl<'dev, Mods: crate::WrappedData> crate::WrappedData for CPU2<'dev, Mods> {
    type Wrap<T, Base: crate::HasId + crate::PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: crate::HasId + crate::PtrType>(
        &self,
        base: Base,
    ) -> Self::Wrap<T, Base> {
        self.modules.wrap_in_base(base)
    }

    #[inline]
    fn wrapped_as_base<T, Base: crate::HasId + crate::PtrType>(
        wrap: &Self::Wrap<T, Base>,
    ) -> &Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: crate::HasId + crate::PtrType>(
        wrap: &mut Self::Wrap<T, Base>,
    ) -> &mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<'a, Mods: OnDropBuffer> Device for CPU2<'a, Mods> {
    type Error = Infallible;
    type Base<T: Unit, S: Shape> = CPUPtr<T>;
    type Data<T: Unit, S: Shape> = Self::Wrap<T, Self::Base<T, S>>;
    // type WrappedData<T, S: Shape> = ;

    fn new() -> Result<Self, Self::Error> {
        todo!()
    }

    #[inline(always)]
    fn base_to_data<T: Unit, S: Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn wrap_to_data<T: Unit, S: Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<T: Unit, S: Shape>(
        data: &Self::Data<T, S>,
    ) -> &Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<T: Unit, S: Shape>(
        data: &mut Self::Data<T, S>,
    ) -> &mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    // #[inline]
    // fn wrap(&self) {}
}

impl<'a, SimpleMods> CPU2<'a, SimpleMods> {
    #[inline]
    pub fn new<NewMods>() -> CPU2<'a, SimpleMods::Module>
    where
        SimpleMods: Module<'a, CPU2<'a>, Module = NewMods>,
        // NewMods: Setup<CPU2<'a, NewMods>>,
    {
        let mut cpu = CPU2 {
            modules: SimpleMods::new(),
            graph: Default::default(),
            pd: PhantomData,
        };
        // NewMods::setup(&mut cpu).unwrap();
        cpu
    }
}

impl<T: Unit, Mods: OnDropBuffer> Alloc<T> for CPU2<'_, Mods> {
    fn alloc<S: Shape>(&self, mut len: usize, flag: AllocFlag) -> custos::Result<Self::Base<T, S>> {
        if len == 0 {
            return Err(DeviceError::ZeroLengthBuffer.into());
        }

        if S::LEN > len {
            len = S::LEN
        }

        Ok(CPUPtr::new_initialized(len, flag))
    }

    fn alloc_from_slice<S>(&self, data: &[T]) -> custos::Result<Self::Base<T, S>>
    where
        S: Shape,
        T: Clone,
    {
        if data.is_empty() {
            return Err(DeviceError::ZeroLengthBuffer.into());
        }
        if !(S::LEN == data.len() || S::LEN == 0) {
            return Err(DeviceError::ShapeLengthMismatch.into());
        }

        let cpu_ptr = unsafe { CPUPtr::new(data.len(), AllocFlag::None) };
        let slice = unsafe { std::slice::from_raw_parts_mut(cpu_ptr.ptr, data.len()) };
        slice.clone_from_slice(data);

        Ok(cpu_ptr)
    }
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
    tape: UnsafeCell<Tape<'a>>,
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
    fn get_grad<T, S>(&self, device: &'dev D, for_buf_id: Id) -> &Buffer<'dev, T, D, S>
    where
        D: Alloc<T>,
        T: 'static,
        S: Shape;

    #[allow(clippy::mut_from_ref)]
    unsafe fn get_grad_mut<'b, T, S>(&'b self, for_buf_id: Id) -> &'b mut Buffer<'dev, T, D, S>
    where
        T: 'static,
        S: Shape;

    fn grad<T, S>(&self, device: &'dev D, for_buf: &Buffer<'_, T, D, S>) -> &Buffer<'dev, T, D, S>
    where
        T: 'static,
        D: Alloc<T>,
        S: Shape,
    {
        self.get_grad(device, for_buf.id())
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
    fn get_grad<'a, T, S>(&'a self, device: &'dev D, for_buf_id: Id) -> &'a Buffer<'dev, T, D, S>
    where
        D: Alloc<T>,
        T: 'static,
        S: Shape,
    {
        let mut new_buf = false;
        unsafe { (*self._cache.get()).add_buf_once::<T, D, S>(device, for_buf_id, &mut new_buf) };
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
    fn get_grad<'a, T, S>(
        &'a self,
        device: &'dev Self,
        for_buf_id: Id,
    ) -> &'a Buffer<'dev, T, Self, S>
    where
        T: 'static,
        S: Shape,
    {
        self.modules.get_grad(device, for_buf_id)
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
            tape: Default::default(),
            _modules: Mods::new(),
            val: Default::default(),
        }
    }
}

impl<'a, 'b, T, D, S, Mods: OnNewBuffer<'a, T, D, S>> OnNewBuffer<'b, T, D, S>
    for Autograd<'a, Mods>
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

impl<'dev, T, D, S> Grad<'dev, T, D, S> for Buffer<'dev, T, D, S>
where
    T: 'static,
    D: Device + 'static + GradActions<'dev, D> + Alloc<T>,
    S: Shape,
{
    fn grad1(&self) -> &Buffer<'dev, T, D, S> {
        self.device().get_grad(self.device(), self.id())
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
    Mods: 'static
        + for<'d> GradActions<'d, Self>
        + OnDropBuffer
        + AddOperation
        + Retrieve<Self, T, S>,
{
    fn add(
        &self,
        lhs: &mut Buffer<T, Self, S>,
        rhs: &mut Buffer<T, Self, S>,
    ) -> Buffer<T, Self, S> {
        let out = self.retrieve(lhs.len, (&*lhs, &*rhs)).unwrap();

        // lazy fn not grad fn -> wurscht
        self.add_op((lhs, rhs /*&out*/), |(lhs, rhs /*out*/)| {
            lhs.grad_mut1();
            rhs.grad1();
            // add_ew_grad_slice(lhs.grad_mut1(), out.grad1());
            // add_ew_grad_slice(rhs.grad_mut1(), out.grad1());
            Ok(())
        })
        .unwrap();

        out
    }

    fn test<'a>(&self, lhs: &'a Buffer<'_, T, Self, S>) -> &'a Buffer<'dev, T, Self, S> {
        todo!()
        // lhs.grad1()
    }
}

fn add_ew_grad_slice<T: Copy + AddAssign>(grad_acc: &mut [T], out_grad: &[T]) {
    for (grad, out_grad) in grad_acc.iter_mut().zip(out_grad) {
        *grad += *out_grad;
    }
}

pub trait OnNewBuffer2<'dev, T: Unit, D: Device + 'dev, S: Shape = ()> {
    fn on_new_buffer2(new_buf: &Buffer<'dev, T, D, S>) {}
    fn on_new_buffer3(&self, x: &Buffer<'dev, T, D, S>) {}
    // fn on_new_buffer2(&self, /*_device: &'dev D,*/ _new_buf: &'_ Buffer<'dev, T, D, S>) {}
}

impl<'dev, Mods: OnNewBuffer2<'dev, T, Self, S> + OnDropBuffer + 'dev, T, S: Shape>
    OnNewBuffer2<'dev, T, Self, S> for CPU<Mods>
{
    // fn on_new_buffer2(&self, _device: &'dev D, _new_buf: &Buffer<'dev, T, D, S>) {
    //     self.modules.on_new_buffer2(_device, _new_buf)
    // }
}

impl<'dev, Mods: OnNewBuffer2<'dev, T, D, S>, T, D: Device + 'dev, S: Shape>
    OnNewBuffer2<'dev, T, D, S> for Autograd<'dev, Mods>
{
    // fn on_new_buffer2(&self, _device: &'dev D, _new_buf: &Buffer<'dev, T, D, S>) {
    //     self._modules.on_new_buffer2(_device, _new_buf)
    // }
}

impl<'dev, Mods: OnNewBuffer2<'dev, T, D, S>, T, D: Device + 'dev, S: Shape, SD: Device>
    OnNewBuffer2<'dev, T, D, S> for CachedModule<Mods, SD>
{
    // fn on_new_buffer2(&self, _device: &'dev D, _new_buf: &Buffer<'dev, T, D, S>) {
    //     self.modules.on_new_buffer2(_device, _new_buf)
    // }
}

impl<'dev, T, D: Device + 'dev, S: Shape> OnNewBuffer2<'dev, T, D, S> for Base {}

fn x<'a>(device: &'a CPU2<'a, Autograd<'a, Base>>) {
    let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
    let rhs = device.buffer([1f32, 2., 6., 4., 5.]);
    // let mut buffers = HashMap::default();
    // // unsafe { register_buf_copyable(&mut buffers, &lhs) };
    // // unsafe { register_buf_copyable(&mut buffers, &rhs) };
    // let tape: &'a mut LazyGraph2<'a> = &mut unsafe { &mut *device.modules.tape.get()}.lazy_graph;
    // tape.add_operation((&lhs, &rhs), |(lhs, rhs)| {
    //     // lhs.grad();
    //     Ok(())
    // });
    // tape.call_lazily(&mut buffers).unwrap();
}
impl<'a, T, S: Shape, Mods: OnDropBuffer> DevicelessAble<'a, T, S> for CPU2<'_, Mods> {}
fn main() {
    // let x = Box::new(Typ::default());
    // Box::into_raw(x);
    //
    {
        // x(&device);
        let mut device = CPU::<custos::Autograd<Base>>::new();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);

        // let lhs = Buffer::<f32, _>::deviceless(&device, 10);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);
        // let rhs = Buffer::<f32, _>::deviceless(&device, 10);
        let mut buffers = HashMap::default();

        // let graph = unsafe { &mut *device.graph.get() };
        // let mut graph = &mut device.graph;

        device.add_grad_fn((&lhs, &rhs), |(lhs, rhs)| {
            unsafe { lhs.grad_mut() };
            Ok(())
        });

        unsafe { device.modules.tape_mut() }
            .unwrap()
            .backward(&mut buffers, false);
        // unsafe { device.modules.tape_mut() }.unwrap().backward_seeded_with_buffers(&lhs, &[1., 1., 1., 1., 1.], &mut buffers);
        rhs.backward();

        // graph.add_operation((&lhs, &rhs), |(lhs, rhs)| Ok(()));
        let graph: &mut LazyGraph2 = &mut unsafe { device.modules.tape_mut() }.unwrap().lazy_graph;
        graph.call_lazily(&mut buffers).unwrap();
        //        // unsafe { register_buf_copyable(&mut buffers, &lhs) };
        // unsafe { register_buf_copyable(&mut buffers, &rhs) };
        // let tape: &mut LazyGraph2 = &mut unsafe { &mut *device.modules.tape.get()}.lazy_graph;
        // tape.add_operation((&lhs, &rhs), |(lhs, rhs)| {
        // lhs.grad();
        // Ok(())
        // });
        // tape.call_lazily(&device, &mut buffers).unwrap();

        let dev = CPU::<Autograd<Cached<Base>>>::new1();

        let data = /*dev.wrap_in_base(*/dev.alloc::<()>(10, custos::flag::AllocFlag::None).unwrap();
        let buffer: Buffer<f32, _> = Buffer { data, device: None };
        // CPU::<Autograd<Base>>::on_new_buffer2(&buffer);
        // dev.on_new_buffer3(&buffer);
        // OnNewBuffer2::<_ ,_>::on_new_buffer3(&dev, &buffer)
        // dev.on_new_buffer2(&buffer)
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

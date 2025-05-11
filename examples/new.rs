use std::cell::{Ref, RefCell};
use std::marker::PhantomData;
use std::{cell::UnsafeCell, convert::Infallible};

use custos::cpu::CPUPtr;
use custos::flag::AllocFlag;
use custos::{
    AddOperation, Alloc, Base, BorrowCacheLT, Buffer, Device, DeviceError, HasId, Id, Module, Num, OnDropBuffer, OnNewBuffer, PtrType, Shape, Unit, WrappedData
};


#[derive(Default)]
pub struct OnlyCaching<'s_oc_dev, Mods> {
    _cache: RefCell<BorrowCacheLT<'s_oc_dev>>,
    _modules: Mods,
}

pub trait BufRetrieve<'m_dev, T, D: Device, S: Shape> {
    fn buf_retrieve(&self, device: &'m_dev D, len: usize)
        -> Ref<Buffer<'m_dev, T, D, S>>;
}

impl<'s_oc_dev, 'm_dev, Mods, T: 'static, D: Device + 'static, S: Shape>
    BufRetrieve<'m_dev, T, D, S> for OnlyCaching<'s_oc_dev, Mods>
where
    's_oc_dev: 'm_dev,
    Mods: 's_oc_dev,
{
    fn buf_retrieve(
        &self,
        device: &'m_dev D,
        len: usize,
    ) -> Ref<Buffer<'m_dev, T, D, S>> {
        // let cache_ref: &mut BorrowCacheLT<'s_oc_dev> = unsafe { &mut *self._cache.get() };
        let cache_ref = self._cache.borrow();
        // cache_ref.add_buf::<T, D, S>(Id { id: 0, len }).unwrap();
        Ref::map(cache_ref, |cache_ref| {
            cache_ref.get_buf(Id { id: 0, len}).unwrap()
        })
        // cache_ref.get_buf::<T, D, S>(Id { id: 0, len }).unwrap()
    }
}

impl<'a, D: 'a, Mods: Module<'a, D>> Module<'a, D> for OnlyCaching<'a, Mods> {
    type Module = OnlyCaching<'a, Mods::Module>;

    fn new() -> Self::Module {
        OnlyCaching {
            _cache: Default::default(),
            _modules: Mods::new(),
        }
    }
}

pub struct CPULT<Mods> {
    mods: Mods,
    // p: PhantomData<&'s_cpu_dev Mods>
}

impl<'s_cpu_dev, 'm_dev, T: 'static, D: Device + 'static, S: Shape, Mods: BufRetrieve<'m_dev, T, D, S>> BufRetrieve<'m_dev, T, D, S>
    for CPULT<Mods>
where
    's_cpu_dev: 'm_dev,
{
    fn buf_retrieve(
        &self,
        device: &'m_dev D,
        len: usize,
    ) -> Ref<Buffer<'m_dev, T, D, S>> {
        self.mods.buf_retrieve(device, len)
    }
}

impl<'dev, T, D, S, Mods> OnNewBuffer<'dev, T, D, S> for CPULT<Mods>
where
    Self: 'dev,
    T: Unit,
    D: Device,
    S: Shape,
    Mods: OnNewBuffer<'dev, T, D, S>,
{
    #[inline]
    fn on_new_buffer(&self, device: &'dev D, new_buf: &Buffer<'dev, T, D, S>) {
        self.mods.on_new_buffer(device, new_buf)
    }
}

impl<T: Unit, Mods> Alloc<T> for CPULT<Mods> {
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

impl<'dev, Mods> WrappedData for CPULT<Mods> {
    type Wrap<T, Base: HasId + PtrType> = Num<T>;

    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        todo!()
    }

    fn wrapped_as_base<T, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base {
        todo!()
    }

    fn wrapped_as_base_mut<T, Base: HasId + PtrType>(wrap: &mut Self::Wrap<T, Base>) -> &mut Base {
        todo!()
    }
}

impl<'dev, Mods> OnDropBuffer for CPULT<Mods> {
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(&self, _device: &D, _buf: &Buffer<T, D, S>) {}
}

impl<'dev, Mods> Device for CPULT<Mods> {
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
}

impl<'a, SimpleMods> CPULT<SimpleMods> {
    #[inline]
    pub fn new<NewMods>() -> CPULT<SimpleMods::Module>
    where
        SimpleMods: Module<'a, CPULT<Base>, Module = NewMods>,
        // NewMods: Setup<CPULT<'a, NewMods>>,
    {
        let mut cpu = CPULT {
            mods: SimpleMods::new(),
            // p: PhantomData,
        };
        // NewMods::setup(&mut cpu).unwrap();
        cpu
    }
}

impl<'a, 'b, T, D, S, Mods: OnNewBuffer<'a, T, D, S>> OnNewBuffer<'b, T, D, S>
    for OnlyCaching<'a, Mods>
where
    D: Device,
    S: Shape,
{
    fn on_new_buffer(&self, _device: &'b D, _new_buf: &Buffer<'b, T, D, S>) {
        // self.
    }
}

impl<'a, Mods: OnDropBuffer> AddOperation for OnlyCaching<'a, Mods> {
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

impl<'dev, Mods> AddOperation for CPULT<Mods> {
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


pub trait AddBuf<'dev, T: Unit, S: Shape = (), D: Device = Self>: Sized + Device {
    fn add(&self, lhs: &Buffer<'dev, T, D, S>, rhs: &Buffer<'dev, T, D, S>) -> Buffer<T, Self, S>;
}


impl<'dev, T: 'static, S: Shape, Mods> AddBuf<'dev, T, S, Self> for CPULT<Mods> 
where 
    Self: 'static,
    Mods: BufRetrieve<'dev, T, Self, S> + AddOperation
{
    fn add(&self, lhs: &Buffer<'dev, T, Self, S>, rhs: &Buffer<'dev, T, Self, S>) -> Buffer<T, Self, S> {
        
        self.add_op((lhs, rhs), |(lhs, rhs)| {
            let dev = lhs.device();
            dev.buf_retrieve(dev, 10);
            // let dev = lhs.device();
            // dev.buf_retrieve(dev, 10);
            Ok(())
        }).unwrap();
        todo!()
    }
}



fn main() {
    let cpu_static = CPULT::<OnlyCaching<Base>>::new();

    let buf = cpu_static.buffer([1, 2, 3, 4]);
    // Buffer::new(&cpu_static, 100);
    let _buf1: Ref<Buffer<i32, _, ()>> = cpu_static.buf_retrieve(&cpu_static, 10);
    cpu_static.add(&buf, &*_buf1);
    // cpu_static.add(&buf, _buf1);
}

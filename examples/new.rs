use std::marker::PhantomData;
use std::{cell::UnsafeCell, convert::Infallible};

use custos::cpu::CPUPtr;
use custos::{
    BorrowCacheLT, Buffer, Device, HasId, Id, Num, OnDropBuffer, PtrType, Shape, Unit, WrappedData,
};

#[derive(Default)]
pub struct Base;

pub struct Puffer<'a, T, D, S> {
    pub ptr: PhantomData<&'a ()>,
    pub s: PhantomData<S>,
    pub d: PhantomData<D>,
    pub t: PhantomData<T>,
}

#[derive(Default)]
pub struct OnlyCaching<'s_oc_dev, Mods> {
    _cache: UnsafeCell<BorrowCacheLT<'s_oc_dev>>,
    _modules: Mods,
}

pub trait BufRetrieve<'m_dev, T, D: Device, S: Shape> {
    fn buf_retrieve(&'m_dev self, device: &'m_dev D, len: usize)
        -> &'m_dev Buffer<'m_dev, T, D, S>;
}

impl<'s_oc_dev, 'm_dev, Mods, T: 'static, D: Device + 'static, S: Shape>
    BufRetrieve<'m_dev, T, D, S> for OnlyCaching<'s_oc_dev, Mods>
where
    's_oc_dev: 'm_dev,
    Mods: 's_oc_dev,
{
    fn buf_retrieve(
        &'m_dev self,
        device: &'m_dev D,
        len: usize,
    ) -> &'m_dev Buffer<'m_dev, T, D, S> {
        let cache_ref: &'m_dev mut BorrowCacheLT<'s_oc_dev> = unsafe { &mut *self._cache.get() };
        // cache_ref.add_buf::<T, D, S>(Id { id: 0, len }).unwrap();
        cache_ref.get_buf::<T, D, S>(Id { id: 0, len }).unwrap()
    }
}

pub struct CPULT<'s_cpu_dev> {
    mods: OnlyCaching<'s_cpu_dev, Base>,
}

impl<'s_cpu_dev, 'm_dev, T: 'static, D: Device + 'static, S: Shape> BufRetrieve<'m_dev, T, D, S>
    for CPULT<'s_cpu_dev>
where
    's_cpu_dev: 'm_dev,
{
    fn buf_retrieve(
        &'m_dev self,
        device: &'m_dev D,
        len: usize,
    ) -> &'m_dev Buffer<'m_dev, T, D, S> {
        self.mods.buf_retrieve(device, len)
    }
}

impl<'dev> WrappedData for CPULT<'dev> {
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

impl<'dev> OnDropBuffer for CPULT<'dev> {
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(&self, _device: &D, _buf: &Buffer<T, D, S>) {}
}

impl<'dev> Device for CPULT<'dev> {
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

fn main() {
    let cpu_static = CPULT {
        mods: Default::default(),
    };

    let _buf1: &Buffer<f32, _, ()> = cpu_static.buf_retrieve(&cpu_static, 10);
}

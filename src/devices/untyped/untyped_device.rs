use core::marker::PhantomData;

use crate::{
    Base, Buffer, Device, HasId, HasModules, OnDropBuffer, PtrType, Shape, WrappedData, CPU,
};

use super::storages::{CpuStorage, CudaStorage, UntypedData};

#[cfg(feature = "cuda")]
pub type Cuda<Mods> = crate::CUDA<Mods>;

#[cfg(not(feature = "cuda"))]
pub type Cuda<Mods> = super::CUDA<Mods>;

pub enum UntypedDevice {
    CPU(CPU<Base>),
    CUDA(Cuda<Base>),
}

pub struct Untyped {
    pub device: UntypedDevice,
}

impl Device for Untyped {
    type Base<T, S: crate::Shape> = UntypedData;
    type Data<T, S: crate::Shape> = UntypedData;
    type Error = ();

    #[inline]
    fn base_to_data<T, S: crate::Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        base
    }

    #[inline]
    fn wrap_to_data<T, S: crate::Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline]
    fn data_as_wrap<'a, T, S: crate::Shape>(
        data: &'a Self::Data<T, S>,
    ) -> &'a Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline]
    fn data_as_wrap_mut<'a, T, S: crate::Shape>(
        data: &'a mut Self::Data<T, S>,
    ) -> &'a mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }
}

impl OnDropBuffer for Untyped {
    #[inline]
    fn on_drop_buffer<T, D: Device, S: Shape>(&self, _device: &D, _buf: &Buffer<T, D, S>) {}
}

impl WrappedData for Untyped {
    type Wrap<T, Base: HasId + PtrType> = Base;

    #[inline]
    fn wrap_in_base<T, Base: crate::HasId + crate::PtrType>(
        &self,
        base: Base,
    ) -> Self::Wrap<T, Base> {
        base
    }

    #[inline]
    fn wrapped_as_base<T, Base: crate::HasId + crate::PtrType>(
        wrap: &Self::Wrap<T, Base>,
    ) -> &Base {
        wrap
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: crate::HasId + crate::PtrType>(
        wrap: &mut Self::Wrap<T, Base>,
    ) -> &mut Base {
        wrap
    }
}

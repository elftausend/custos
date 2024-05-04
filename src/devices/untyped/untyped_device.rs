use crate::{
    Alloc, Base, Buffer, Device, HasId, HasModules, IsShapeIndep, OnDropBuffer, OnNewBuffer,
    PtrType, Retriever, Shape, WrappedData, CPU,
};

use super::{
    storages::{CpuStorage, CudaStorage, UntypedData},
    AsType,
};

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
    type Error = crate::Error;

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

    #[inline]
    fn new() -> Result<Self, Self::Error> {
        Ok(Untyped {
            #[cfg(not(feature = "cuda"))]
            device: UntypedDevice::CPU(CPU::based()),
            #[cfg(feature = "cuda")]
            device: UntypedDevice::CUDA(crate::CUDA::<Base>::new(crate::cuda::chosen_cu_idx())?),
        })
    }
}

unsafe impl IsShapeIndep for Untyped {}

impl HasModules<Base> for Untyped {
    #[inline]
    fn modules(&self) -> &Base {
        match &self.device {
            UntypedDevice::CPU(cpu) => &cpu.modules,
            UntypedDevice::CUDA(cuda) => &cuda.modules,
        }
    }
}

impl OnDropBuffer for Untyped {}
impl<T, D: Device, S: Shape> OnNewBuffer<T, D, S> for Untyped {}

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

impl<T: AsType> Alloc<T> for Untyped {
    fn alloc<S: Shape>(
        &self,
        len: usize,
        flag: crate::flag::AllocFlag,
    ) -> crate::Result<Self::Base<T, S>> {
        Ok(match &self.device {
            UntypedDevice::CPU(cpu) => {
                UntypedData::CPU(CpuStorage::from(Alloc::<T>::alloc::<S>(cpu, len, flag)?))
            }
            UntypedDevice::CUDA(cuda) => {
                #[cfg(feature = "cuda")]
                {
                    UntypedData::CUDA(CudaStorage::from(Alloc::<T>::alloc::<S>(cuda, len, flag)?))
                }
                #[cfg(not(feature = "cuda"))]
                unimplemented!()
            }
        })
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        Ok(match &self.device {
            UntypedDevice::CPU(cpu) => UntypedData::CPU(CpuStorage::from(
                Alloc::<T>::alloc_from_slice::<S>(cpu, data)?,
            )),
            UntypedDevice::CUDA(cuda) => {
                #[cfg(feature = "cuda")]
                {
                    UntypedData::CUDA(CudaStorage::from(Alloc::<T>::alloc_from_slice::<S>(
                        cuda, data,
                    )?))
                }

                #[cfg(not(feature = "cuda"))]
                unimplemented!()
            }
        })
    }
}

impl<T: AsType, S: Shape> Retriever<T, S> for Untyped {
    #[inline]
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        len: usize,
        _parents: impl crate::Parents<NUM_PARENTS>,
    ) -> crate::Result<Buffer<T, Self, S>> {
        let data = Alloc::<T>::alloc::<S>(self, len, crate::flag::AllocFlag::None)?;
        Ok(Buffer {
            data,
            device: Some(self),
        })
    }
}

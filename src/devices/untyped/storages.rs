mod cpu_storage;
use core::mem::transmute;

pub use cpu_storage::*;

mod cuda_storage;
pub use cuda_storage::*;

use crate::{untyped::DeviceType, Device, HasId, PtrType, Shape};

use super::{AsDeviceType, AsType, MatchesType};

pub enum UntypedData {
    CPU(CpuStorage),
    CUDA(CudaStorage),
}

impl PtrType for UntypedData {
    #[inline]
    fn size(&self) -> usize {
        match self {
            UntypedData::CPU(cpu) => cpu.size(),
            UntypedData::CUDA(cuda) => cuda.size(),
        }
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        match self {
            UntypedData::CPU(cpu) => cpu.flag(),
            UntypedData::CUDA(cuda) => cuda.flag(),
        }
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        match self {
            UntypedData::CPU(cpu) => cpu.set_flag(flag),
            UntypedData::CUDA(cuda) => cuda.set_flag(flag),
        }
    }
}

impl HasId for UntypedData {
    #[inline]
    fn id(&self) -> crate::Id {
        match self {
            UntypedData::CPU(cpu) => cpu.id(),
            UntypedData::CUDA(cuda) => cuda.id(),
        }
    }
}

impl MatchesType for UntypedData {
    #[inline]
    fn matches_storage_type<T: super::AsType>(&self) -> Result<(), String> {
        match self {
            UntypedData::CPU(cpu) => cpu.matches_storage_type::<T>(),
            UntypedData::CUDA(cuda) => cuda.matches_storage_type::<T>(),
        }
    }
}

impl UntypedData {
    pub fn convert_to_typed<T: AsType, D: AsDeviceType + Device, S: Shape>(
        &self,
    ) -> Option<&D::Base<T, S>> {
        self.matches_storage_type::<T>().ok()?;
        match self {
            UntypedData::CPU(cpu) => {
                if D::DEVICE_TYPE != DeviceType::CPU {
                    return None;
                }
                Some(match cpu {
                    CpuStorage::U8(data) => unsafe { transmute(data) },
                    CpuStorage::U32(data) => unsafe { transmute(data) },
                    CpuStorage::I64(data) => unsafe { transmute(data) },
                    CpuStorage::BF16(data) => unsafe { transmute(data) },
                    CpuStorage::F16(data) => unsafe { transmute(data) },
                    CpuStorage::F32(data) => unsafe { transmute(data) },
                    CpuStorage::F64(data) => unsafe { transmute(data) },
                })
            }
            UntypedData::CUDA(cuda) => {
                if D::DEVICE_TYPE != DeviceType::CUDA {
                    return None;
                }
                #[cfg(feature = "cuda")]
                {
                    Some(match cuda {
                        CudaStorage::U8(data) => unsafe { transmute(data) },
                        CudaStorage::U32(data) => unsafe { transmute(data) },
                        CudaStorage::I64(data) => unsafe { transmute(data) },
                        CudaStorage::BF16(data) => unsafe { transmute(data) },
                        CudaStorage::F16(data) => unsafe { transmute(data) },
                        CudaStorage::F32(data) => unsafe { transmute(data) },
                        CudaStorage::F64(data) => unsafe { transmute(data) },
                    })
                }

                #[cfg(not(feature = "cuda"))]
                None
            }
        }
    }
}

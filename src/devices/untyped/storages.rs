mod cpu_storage;
use core::mem::transmute;

pub use cpu_storage::*;

mod cuda_storage;
pub use cuda_storage::*;

use crate::{Device, HasId, PtrType, Shape, untyped::DeviceType};

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

macro_rules! convert_to_typed {
    ($self:ident) => {
        match $self {
            UntypedData::CPU(cpu) => {
                if D::DEVICE_TYPE != DeviceType::CPU {
                    return None;
                }
                Some(match cpu {
                    CpuStorage::U8(data) => unsafe { transmute(data) },
                    CpuStorage::U32(data) => unsafe { transmute(data) },
                    CpuStorage::I64(data) => unsafe { transmute(data) },
                    #[cfg(feature = "half")]
                    CpuStorage::BF16(data) => unsafe { transmute(data) },
                    #[cfg(feature = "half")]
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
                        #[cfg(feature = "half")]
                        CudaStorage::BF16(data) => unsafe { transmute(data) },
                        #[cfg(feature = "half")]
                        CudaStorage::F16(data) => unsafe { transmute(data) },
                        CudaStorage::F32(data) => unsafe { transmute(data) },
                        CudaStorage::F64(data) => unsafe { transmute(data) },
                    })
                }

                #[cfg(not(feature = "cuda"))]
                None
            }
        }
    };
}

impl UntypedData {
    pub fn convert_to_typed<T: AsType, D: AsDeviceType + Device, S: Shape>(
        &self,
    ) -> Option<&D::Base<T, S>> {
        self.matches_storage_type::<T>().ok()?;
        convert_to_typed!(self)
    }
    // add "checked" to name maybe, then add unsafe variants..
    pub fn convert_to_typed_mut<T: AsType, D: AsDeviceType + Device, S: Shape>(
        &mut self,
    ) -> Option<&mut D::Base<T, S>> {
        self.matches_storage_type::<T>().ok()?;
        convert_to_typed!(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{CPU, cpu::CPUPtr};

    use super::{CpuStorage, UntypedData};

    #[test]
    fn test_convert_untyped_to_typed_checked() {
        let data = UntypedData::CPU(CpuStorage::U32(CPUPtr::new_initialized(
            10,
            crate::flag::AllocFlag::None,
        )));
        data.convert_to_typed::<u32, CPU, ()>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_convert_untyped_to_typed_checked_mismatching_data_types() {
        let data = UntypedData::CPU(CpuStorage::U32(CPUPtr::new_initialized(
            10,
            crate::flag::AllocFlag::None,
        )));
        data.convert_to_typed::<f32, CPU, ()>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_convert_untyped_to_typed_checked_mismatching_device_types() {
        use crate::untyped::untyped_device::Cuda;

        let data = UntypedData::CPU(CpuStorage::U32(CPUPtr::new_initialized(
            10,
            crate::flag::AllocFlag::None,
        )));
        data.convert_to_typed::<u32, Cuda<crate::Base>, ()>()
            .unwrap();
    }
}

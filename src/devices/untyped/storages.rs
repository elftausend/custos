mod cpu_storage;
pub use cpu_storage::*;

mod cuda_storage;
pub use cuda_storage::*;

use crate::{Device, HasId, PtrType, Shape};

use super::MatchesType;

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
            UntypedData::CUDA(cuda) => cuda.id()
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
    pub fn convert_to_typed<T, D: Device, S: Shape>(&self) -> &D::Data<T, S> {
        match self {
            UntypedData::CPU(cpu) => {
                
                todo!()
            },
            UntypedData::CUDA(cuda) => todo!(),
        }
    }
}

use crate::{cpu::CPUPtr, HasId, PtrType};

#[derive(Debug)]
pub enum CpuStorage {
    U8(CPUPtr<u8>),
    U32(CPUPtr<u32>),
    I64(CPUPtr<i64>),
    #[cfg(feature = "half")]
    BF16(CPUPtr<half::bf16>),
    #[cfg(feature = "half")]
    F16(CPUPtr<half::f16>),
    F32(CPUPtr<f32>),
    F64(CPUPtr<f64>),
}

impl PtrType for CpuStorage {
    #[inline]
    fn size(&self) -> usize {
        match self {
            CpuStorage::U8(ptr) => ptr.size(),
            CpuStorage::U32(ptr) => ptr.size(),
            CpuStorage::I64(ptr) => ptr.size(),
            #[cfg(feature = "half")]
            CpuStorage::BF16(ptr) => ptr.size(),
            #[cfg(feature = "half")]
            CpuStorage::F16(ptr) => ptr.size(),
            CpuStorage::F32(ptr) => ptr.size(),
            CpuStorage::F64(ptr) => ptr.size(),
        }
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        match self {
            CpuStorage::U8(ptr) => ptr.flag(),
            CpuStorage::U32(ptr) => ptr.flag(),
            CpuStorage::I64(ptr) => ptr.flag(),
            #[cfg(feature = "half")]
            CpuStorage::BF16(ptr) => ptr.flag(),
            #[cfg(feature = "half")]
            CpuStorage::F16(ptr) => ptr.flag(),
            CpuStorage::F32(ptr) => ptr.flag(),
            CpuStorage::F64(ptr) => ptr.flag(),
        }
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        match self {
            CpuStorage::U8(ptr) => ptr.set_flag(flag),
            CpuStorage::U32(ptr) => ptr.set_flag(flag),
            CpuStorage::I64(ptr) => ptr.set_flag(flag),
            #[cfg(feature = "half")]
            CpuStorage::BF16(ptr) => ptr.set_flag(flag),
            #[cfg(feature = "half")]
            CpuStorage::F16(ptr) => ptr.set_flag(flag),
            CpuStorage::F32(ptr) => ptr.set_flag(flag),
            CpuStorage::F64(ptr) => ptr.set_flag(flag),
        }
    }
}

impl HasId for CpuStorage {
    #[inline]
    fn id(&self) -> crate::Id {
        match self {
            CpuStorage::U8(ptr) => ptr.id(),
            CpuStorage::U32(ptr) => ptr.id(),
            CpuStorage::I64(ptr) => ptr.id(),
            #[cfg(feature = "half")]
            CpuStorage::BF16(ptr) => ptr.id(),
            #[cfg(feature = "half")]
            CpuStorage::F16(ptr) => ptr.id(),
            CpuStorage::F32(ptr) => ptr.id(),
            CpuStorage::F64(ptr) => ptr.id(),
        }
    }
}


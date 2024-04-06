use crate::{
    cpu::CPUPtr,
    untyped::{AsType, Type},
    HasId, PtrType,
};

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

impl crate::untyped::MatchesType for CpuStorage {
    fn matches_storage_type<T: AsType>(&self) -> Result<(), String> {
        match (T::TYPE, self) {
            (Type::U8, CpuStorage::U8(_)) => Ok(()),
            (Type::U32, CpuStorage::U32(_)) => Ok(()),
            (Type::I64, CpuStorage::I64(_)) => Ok(()),
            (Type::BF16, CpuStorage::BF16(_)) => Ok(()),
            (Type::F16, CpuStorage::F16(_)) => Ok(()),
            (Type::F32, CpuStorage::F32(_)) => Ok(()),
            (Type::F64, CpuStorage::F64(_)) => Ok(()),
            _ => Err("Storage type mismatch".into()),
        }
    }
}

impl<T: AsType> From<CPUPtr<T>> for CpuStorage {
    fn from(data: CPUPtr<T>) -> Self {
        match T::TYPE {
            Type::U8 => CpuStorage::U8(unsafe { std::mem::transmute(data) }),
            Type::U32 => CpuStorage::U32(unsafe { std::mem::transmute(data) }),
            Type::I64 => CpuStorage::I64(unsafe { std::mem::transmute(data) }),
            Type::BF16 => CpuStorage::BF16(unsafe { std::mem::transmute(data) }),
            Type::F16 => CpuStorage::F16(unsafe { std::mem::transmute(data) }),
            Type::F32 => CpuStorage::F32(unsafe { std::mem::transmute(data) }),
            Type::F64 => CpuStorage::F64(unsafe { std::mem::transmute(data) }),
        }
    }
}

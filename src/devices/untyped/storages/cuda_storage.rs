#[cfg(feature = "cuda")]
use crate::cuda::CUDAPtr;
use crate::{HasId, PtrType};

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub enum CudaStorage {
    U8(CUDAPtr<u8>),
    U32(CUDAPtr<u32>),
    I64(CUDAPtr<i64>),
    #[cfg(feature = "half")]
    BF16(CUDAPtr<half::bf16>),
    #[cfg(feature = "half")]
    F16(CUDAPtr<half::f16>),
    F32(CUDAPtr<f32>),
    F64(CUDAPtr<f64>),
}

#[cfg(feature = "cuda")]
impl PtrType for CudaStorage {
    #[inline]
    fn size(&self) -> usize {
        match self {
            CudaStorage::U8(ptr) => ptr.size(),
            CudaStorage::U32(ptr) => ptr.size(),
            CudaStorage::I64(ptr) => ptr.size(),
            #[cfg(feature = "half")]
            CudaStorage::BF16(ptr) => ptr.size(),
            #[cfg(feature = "half")]
            CudaStorage::F16(ptr) => ptr.size(),
            CudaStorage::F32(ptr) => ptr.size(),
            CudaStorage::F64(ptr) => ptr.size(),
        }
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        match self {
            CudaStorage::U8(ptr) => ptr.flag(),
            CudaStorage::U32(ptr) => ptr.flag(),
            CudaStorage::I64(ptr) => ptr.flag(),
            #[cfg(feature = "half")]
            CudaStorage::BF16(ptr) => ptr.flag(),
            #[cfg(feature = "half")]
            CudaStorage::F16(ptr) => ptr.flag(),
            CudaStorage::F32(ptr) => ptr.flag(),
            CudaStorage::F64(ptr) => ptr.flag(),
        }
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        match self {
            CudaStorage::U8(ptr) => ptr.set_flag(flag),
            CudaStorage::U32(ptr) => ptr.set_flag(flag),
            CudaStorage::I64(ptr) => ptr.set_flag(flag),
            #[cfg(feature = "half")]
            CudaStorage::BF16(ptr) => ptr.set_flag(flag),
            #[cfg(feature = "half")]
            CudaStorage::F16(ptr) => ptr.set_flag(flag),
            CudaStorage::F32(ptr) => ptr.set_flag(flag),
            CudaStorage::F64(ptr) => ptr.set_flag(flag),

        }
    }
}

impl HasId for CudaStorage {
    #[inline]
    fn id(&self) -> crate::Id {
        match self {
            CudaStorage::U8(ptr) => ptr.id(),
            CudaStorage::U32(ptr) => ptr.id(),
            CudaStorage::I64(ptr) => ptr.id(),
            #[cfg(feature = "half")]
            CudaStorage::BF16(ptr) => ptr.id(),
            #[cfg(feature = "half")]
            CudaStorage::F16(ptr) => ptr.id(),
            CudaStorage::F32(ptr) => ptr.id(),
            CudaStorage::F64(ptr) => ptr.id(),
        }
    }
}


#[cfg(not(feature = "cuda"))]
pub enum CudaStorage {}

#[cfg(not(feature = "cuda"))]
impl PtrType for CudaStorage {
    fn size(&self) -> usize {
        unimplemented!()
    }

    fn flag(&self) -> crate::flag::AllocFlag {
        unimplemented!()
    }

    unsafe fn set_flag(&mut self, _flag: crate::flag::AllocFlag) {
        unimplemented!()
    }
}


#[cfg(not(feature = "cuda"))]
impl HasId for CudaStorage {
    fn id(&self) -> crate::Id {
        unimplemented!()
    }
}

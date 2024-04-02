use crate::{cpu::CPUPtr, cuda::CUDAPtr, PtrType};

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
        todo!()
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        todo!()
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        todo!()
    }
}

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
    fn size(&self) -> usize {
        todo!()
    }

    fn flag(&self) -> crate::flag::AllocFlag {
        todo!()
    }

    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        todo!()
    }
}


#[cfg(not(feature = "cuda"))]
pub enum CudaStorage {}

#[cfg(not(feature = "cuda"))]
impl PtrType for CudaStorage {
    fn size(&self) -> usize {
        todo!()
    }

    fn flag(&self) -> crate::flag::AllocFlag {
        todo!()
    }

    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        todo!()
    }
}
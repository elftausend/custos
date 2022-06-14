#![allow(non_camel_case_types, dead_code)]

use crate::cuda::api::CUstream_st;

use super::error::{CublasResult, CublasErrorKind};

pub enum cublasContext { }
pub type cublasHandle_t = *mut cublasContext;

#[derive(Debug)]
#[repr(C)]
pub enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16,
}

impl cublasStatus_t {
    pub fn to_result(self) -> CublasResult<()> {
        match self {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(CublasErrorKind::from(self as u32))
        }
    }
}


#[repr(u32)]
pub enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2, }

#[link(name="cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: *mut CUstream_st) -> cublasStatus_t;
    pub fn cublasSgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t,
        transb: cublasOperation_t, m: i32,
        n: i32, k: i32,
        alpha: *const f32,
        A: *const f32, lda: i32,
        B: *const f32, ldb: i32,
        beta: *const f32,
        C: *mut f32, ldc: i32)
    -> cublasStatus_t;
    pub fn cublasDgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t,
            transb: cublasOperation_t, m: i32,
            n: i32, k: i32,
            alpha: *const f64,
            A: *const f64, lda: i32,
            B: *const f64, ldb: i32,
            beta: *const f64,
            C: *mut f64, ldc: i32)
    -> cublasStatus_t;
}
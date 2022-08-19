//! This module defines all available compute devices

use self::cpu::{
    api::{cblas_dgemm, cblas_sgemm},
    Order, Transpose,
};
use crate::number::{Float, Number};
use std::cell::RefCell;

#[cfg(feature = "cuda")]
use cuda::api::cublas::{cublasDgemm_v2, cublasOperation_t, cublasSgemm_v2, CublasHandle};

pub mod cache;
pub use cache::Cache;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;

pub type CUdeviceptr = std::os::raw::c_ulonglong;

#[cfg(not(feature = "opencl"))]
#[derive(Debug)]
pub struct InternCLDevice;

#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct InternCudaDevice;

thread_local! {
    pub static COUNT: RefCell<usize> = RefCell::new(0);
}

/// Sets current cache identifier / index.
/// This function is usually called after an iteration in a loop -> [Count](crate::Count) or [range](crate::range)
#[inline]
pub fn set_count(count: usize) {
    COUNT.with(|c| *c.borrow_mut() = count);
}

/// Returns current cache identifier / index
#[inline]
pub fn get_count() -> usize {
    COUNT.with(|c| *c.borrow())
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// A Node is used to identify a cached pointer.
pub struct Node {
    pub idx: usize,
    pub len: usize,
}

impl Node {
    pub fn new(len: usize) -> Node {
        crate::COUNT.with(|count| {
            let node = Node {
                idx: *count.borrow(),
                len,
            };
            *count.borrow_mut() += 1;
            node
        })
    }
}

/// enables easy generic kernel creation
pub trait CDatatype: Number + 'static {
    fn as_c_type_str() -> &'static str;
}

#[cfg(any(not(target_os = "macos"), not(feature = "opencl")))]
impl CDatatype for f64 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "double"
    }
}

#[cfg(any(not(target_os = "macos"), not(feature = "opencl")))]
#[cfg(feature="half")]
impl CDatatype for half::f16 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "half"
    }
}

#[cfg(any(not(target_os = "macos"), not(feature = "opencl")))]
#[cfg(feature="half")]
impl CDatatype for half::bf16 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "half"
    }
}

impl CDatatype for f32 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "float"
    }
}

impl CDatatype for i32 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "int"
    }
}

impl CDatatype for u32 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "uint"
    }
}

impl CDatatype for i8 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "char"
    }
}

impl CDatatype for u8 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "uchar"
    }
}

impl CDatatype for i16 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "short"
    }
}
impl CDatatype for u16 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "ushort"
    }
}

impl CDatatype for i64 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "long"
    }
}

impl CDatatype for u64 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "ulong"
    }
}

pub trait GenericBlas
where
    Self: Sized + Float,
{
    fn blas_gemm(
        order: Order,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    );
    #[inline]
    fn gemm(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        Self::blas_gemm(
            Order::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            a,
            k,
            b,
            n,
            c,
            n,
        )
    }
    #[inline]
    #[allow(non_snake_case)]
    fn gemmT(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        Self::blas_gemm(
            Order::RowMajor,
            Transpose::NoTrans,
            Transpose::Trans,
            m,
            n,
            k,
            a,
            k,
            b,
            k,
            c,
            n,
        )
    }

    #[inline]
    #[allow(non_snake_case)]
    fn Tgemm(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        Self::blas_gemm(
            Order::RowMajor,
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            a,
            m,
            b,
            n,
            c,
            n,
        )
    }

    #[cfg(feature = "cuda")]
    fn cugemm(
        handle: &CublasHandle,
        m: usize,
        n: usize,
        k: usize,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> crate::Result<()>;
}

impl GenericBlas for f32 {
    #[inline]
    fn blas_gemm(
        order: Order,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        unsafe {
            cblas_sgemm(
                order,
                trans_a,
                trans_b,
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                0.0,
                c.as_mut_ptr(),
                ldc,
            )
        };
    }
    #[cfg(feature = "cuda")]
    #[inline]
    fn cugemm(
        handle: &CublasHandle,
        m: usize,
        n: usize,
        k: usize,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> crate::Result<()> {
        unsafe {
            cublasSgemm_v2(
                handle.0,
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &1f32 as *const f32,
                b as *const u64 as *const f32,
                n as i32,
                a as *const u64 as *const f32,
                k as i32,
                &0f32 as *const f32,
                c as *mut u64 as *mut f32,
                n as i32,
            )
        }
        .to_result()?;
        Ok(())
    }
}

impl GenericBlas for f64 {
    #[inline]
    fn blas_gemm(
        order: Order,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) {
        unsafe {
            cblas_dgemm(
                order,
                trans_a,
                trans_b,
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                0.0,
                c.as_mut_ptr(),
                ldc,
            )
        };
    }
    #[cfg(feature = "cuda")]
    #[inline]
    fn cugemm(
        handle: &CublasHandle,
        m: usize,
        n: usize,
        k: usize,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> crate::Result<()> {
        unsafe {
            cublasDgemm_v2(
                handle.0,
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &1f64 as *const f64,
                b as *const u64 as *const f64,
                n as i32,
                a as *const u64 as *const f64,
                k as i32,
                &0f64 as *const f64,
                c as *mut u64 as *mut f64,
                n as i32,
            )
        }
        .to_result()?;
        Ok(())
    }
}

pub fn remove_value<T: Ord>(values: &mut Vec<T>, match_value: &T) -> Result<(), usize> {
    let idx = values.binary_search(match_value)?;
    values.swap_remove(idx);
    Ok(())
}

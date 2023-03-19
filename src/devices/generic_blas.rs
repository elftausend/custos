/// Provides generic access to f32 and f64 BLAS functions

#[cfg(feature = "blas")]
#[cfg(feature = "cpu")]
use super::cpu::{
    api::{cblas_dgemm, cblas_sgemm},
    Order, Transpose,
};

#[cfg(feature = "cuda")]
use super::cuda::api::{
    cublas::{cublasDgemm_v2, cublasOperation_t, cublasSgemm_v2, CublasHandle},
    CUdeviceptr,
};

pub trait GenericBlas
where
    Self: Sized,
{
    #[cfg(feature = "blas")]
    #[cfg(feature = "cpu")]
    #[allow(clippy::too_many_arguments)]
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
    #[cfg(feature = "blas")]
    #[cfg(feature = "cpu")]
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
    #[cfg(feature = "blas")]
    #[cfg(feature = "cpu")]
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

    #[cfg(feature = "blas")]
    #[cfg(feature = "cpu")]
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
    #[cfg(feature = "blas")]
    #[cfg(feature = "cpu")]
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
    #[cfg(feature = "blas")]
    #[cfg(feature = "cpu")]
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

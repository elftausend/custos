//! This module defines all available compute devices

#[cfg(feature = "blas")]
#[cfg(feature = "cpu")]
use self::cpu::{
    api::{cblas_dgemm, cblas_sgemm},
    Order, Transpose,
};
use crate::{shape::Shape, AddGraph, Alloc, Buffer, Device, PtrType};

#[cfg(feature = "cuda")]
use cuda::api::cublas::{cublasDgemm_v2, cublasOperation_t, cublasSgemm_v2, CublasHandle};

#[cfg(not(feature = "no-std"))]
pub mod cache;

#[cfg(not(feature = "no-std"))]
pub mod borrowing_cache;

//pub mod cache;
#[cfg(not(feature = "no-std"))]
pub use cache::*;

//pub use cache::{Cache, CacheReturn};

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "stack")]
pub mod stack;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "network")]
pub mod network;

mod stack_array;
pub use stack_array::*;

mod cdatatype;
pub use cdatatype::*;

mod cpu_stack_ops;

#[cfg(not(feature = "no-std"))]
mod ident;
#[cfg(not(feature = "no-std"))]
pub use ident::*;

#[cfg(feature = "cuda")]
pub type CUdeviceptr = core::ffi::c_ulonglong;

pub trait CacheAble<D: Device> {
    fn retrieve<T, S: Shape>(device: &D, len: usize, add_node: impl AddGraph) -> Buffer<T, D, S>
    where
        for<'a> D: Alloc<'a, T, S>;

    fn get_existing_buf<T, S: Shape>(device: &D, id: Ident) -> Buffer<T, D, S>;

    fn remove(device: &D, ident: Ident);
    fn add_to_cache<T, S: Shape>(device: &D, ptr: &D::Ptr<T, S>) -> Ident;
    //fn insert_node<T>(&mut self, device: &D, ptr: &D::Ptr<T, N>, node: Ident, graph_node: crate::Node) {}
}

// TODO: Mind num implement?
impl<D: Device> CacheAble<D> for () {
    #[inline]
    fn retrieve<T, S: Shape>(device: &D, len: usize, _add_node: impl AddGraph) -> Buffer<T, D, S>
    where
        for<'a> D: Alloc<'a, T, S>,
    {
        Buffer::new(device, len)
    }

    #[inline]
    fn remove(_device: &D, _ident: Ident) {}

    #[inline]
    fn add_to_cache<T, S: Shape>(_device: &D, ptr: &<D as Device>::Ptr<T, S>) -> Ident {
        Ident::new_bumped(ptr.len())
    }

    fn get_existing_buf<T, S: Shape>(device: &D, id: Ident) -> Buffer<T, D, S> {
        todo!();
    }
}

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

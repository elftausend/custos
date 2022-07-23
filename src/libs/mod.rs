use self::cpu::{level3, Order, Transpose, CPU_CACHE};
use crate::number::{Float, Number};
use std::cell::RefCell;

#[cfg(feature = "cuda")]
use cuda::api::cublas::{cublasDgemm_v2, cublasOperation_t, cublasSgemm_v2, CublasHandle};

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
    /// Using a common device count to keep track on device creations.
    /// These device creations are used to know when to deallocate the cached memory.
    /// A seperate count for each device type could be used, however this is a problem for nested device creations.
    /// (Especially for unified memory and device switching)
    pub static DEVICE_COUNT: RefCell<usize> = RefCell::new(0);
}

#[inline]
pub fn get_device_count() -> *mut usize {
    DEVICE_COUNT.with(|c| c.as_ptr())
}

pub fn deallocate_cache(device_count: usize) {
    if device_count != 0 {
        return;
    }

    CPU_CACHE.with(|cache| cache.borrow_mut().nodes.clear());

    #[cfg(feature = "opencl")]
    crate::opencl::CL_CACHE.with(|cache| {
        /*
        // FIXME: releases all kernels, even if it is used by another device?
        // TODO: better kernel cache release
        for kernel in &mut cache.borrow_mut().arg_kernel_cache.values_mut() {
            kernel.release()
        }
        */
        cache.borrow_mut().nodes.clear();
    });

    #[cfg(feature = "cuda")]
    crate::cuda::CUDA_CACHE.with(|cache| cache.borrow_mut().nodes.clear());
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
    fn as_c_type_str() -> &'static str {
        "double"
    }
}

impl CDatatype for f32 {
    fn as_c_type_str() -> &'static str {
        "float"
    }
}

impl CDatatype for i32 {
    fn as_c_type_str() -> &'static str {
        "int"
    }
}

impl CDatatype for u32 {
    fn as_c_type_str() -> &'static str {
        "uint"
    }
}

impl CDatatype for i8 {
    fn as_c_type_str() -> &'static str {
        "char"
    }
}

impl CDatatype for u8 {
    fn as_c_type_str() -> &'static str {
        "uchar"
    }
}

impl CDatatype for i16 {
    fn as_c_type_str() -> &'static str {
        "short"
    }
}
impl CDatatype for u16 {
    fn as_c_type_str() -> &'static str {
        "ushort"
    }
}

impl CDatatype for i64 {
    fn as_c_type_str() -> &'static str {
        "long"
    }
}

impl CDatatype for u64 {
    fn as_c_type_str() -> &'static str {
        "ulong"
    }
}

pub trait GenericBlas
where
    Self: Sized + Float,
{
    fn gemm(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]);
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
    fn gemm(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        unsafe {
            level3::cblas_sgemm(
                Order::RowMajor,
                Transpose::NoTranspose,
                Transpose::NoTranspose,
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
            )
        };
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
    fn gemm(m: usize, n: usize, k: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        unsafe {
            level3::cblas_dgemm(
                Order::RowMajor,
                Transpose::NoTranspose,
                Transpose::NoTranspose,
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
            )
        };
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

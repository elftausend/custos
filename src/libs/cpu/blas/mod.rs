pub mod level3;
pub use level3::*;

pub mod level1;
pub use level1::*;

mod values;
pub use values::*;

pub trait TBlas where Self: Sized {
    fn gemm(m: usize, n: usize, k:usize, a: &[Self], lda: usize, b: &[Self], ldb: usize, c: &mut [Self], ldc: usize);
}

impl TBlas for f32 {
    fn gemm(m: usize, n: usize, k:usize, a: &[Self], lda: usize, b: &[Self], ldb: usize, c: &mut [Self], ldc: usize) {
        unsafe {level3::cblas_sgemm(Order::RowMajor, Transpose::NoTranspose, Transpose::NoTranspose, m, n, k, 1.0, a.as_ptr(), lda, b.as_ptr(), ldb, 0.0, c.as_mut_ptr(), ldc)};
    }
}

impl TBlas for f64 {
    fn gemm(m: usize, n: usize, k:usize, a: &[Self], lda: usize, b: &[Self], ldb: usize, c: &mut [Self], ldc: usize) {
        unsafe {level3::cblas_dgemm(Order::RowMajor, Transpose::NoTranspose, Transpose::NoTranspose, m, n, k, 1.0, a.as_ptr(), lda, b.as_ptr(), ldb, 0.0, c.as_mut_ptr(), ldc)};
    }
}

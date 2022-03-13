pub mod level3;
pub use level3::*;

pub mod level1;
pub use level1::*;

mod values;
pub use values::*;

pub trait TBlas where Self: Sized {
    fn gemm(m: usize, n: usize, k:usize, a: &[Self], b: &[Self], c: &mut [Self]);
}

impl TBlas for f32 {
    fn gemm(m: usize, n: usize, k:usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        unsafe {level3::cblas_sgemm(Order::RowMajor, Transpose::NoTranspose, Transpose::NoTranspose, m, n, k, 1.0, a.as_ptr(), k, b.as_ptr(), n, 0.0, c.as_mut_ptr(), n)};
    }
}

impl TBlas for f64 {
    fn gemm(m: usize, n: usize, k:usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        unsafe {level3::cblas_dgemm(Order::RowMajor, Transpose::NoTranspose, Transpose::NoTranspose, m, n, k, 1.0, a.as_ptr(), k, b.as_ptr(), n, 0.0, c.as_mut_ptr(), n)};
    }
}

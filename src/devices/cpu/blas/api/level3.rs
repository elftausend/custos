//#[cfg_attr(link(name = "BLAS", kind = "framework"))]
//#[cfg_attr(target_os = "windows", link(name = "BLAS"))] ??

use crate::devices::cpu::{Order, Transpose};

#[link(name = "blas")]
//#[cfg_attr(target_os = "macos", link(name = "blas"))]
//#[cfg_attr(target_os = "linux", link(name = "openblas"))]
extern "C" {

    pub fn cblas_sgemm(
        order: Order,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    );

    pub fn cblas_dgemm(
        order: Order,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    );
}

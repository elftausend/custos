
#[link(name="blas")]
extern "C" {
    pub fn cblas_sdot(
        n: u32, alpha: f32, x: *const f32, 
        incx: u32, y: *const f32, incy: u32
    );
    pub fn cblas_saxpy(
        n: usize, alpha: f32, x: *const f32,
        incx: usize, y: *mut f32, incy: usize
    );
    pub fn cblas_sscal(
        n: usize, alpha: f32, x: *mut f32, incx: usize
    );
    pub fn cblas_daxpy(
        n: usize, alpha: f64, x: *const f64,
        incx: usize, y: *mut f64, incy: usize
    );
    pub fn cblas_dscal(
        n: usize, alpha: f64, x: *mut f64, incx: usize
    );
}
//! CUBLAS ffi module

mod error;
mod ffi;

pub use ffi::*;

use self::error::CublasResult;

/// Raw CUBLAS handle
#[derive(Debug)]
pub struct CublasHandle(pub *mut cublasContext);

/// create a CUBLAS handle
pub fn create_handle() -> CublasResult<CublasHandle> {
    let mut handle: CublasHandle = CublasHandle(std::ptr::null_mut());
    unsafe { cublasCreate_v2(&mut handle.0) }.to_result()?;
    Ok(handle)
}

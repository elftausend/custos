mod ffi;
mod error;

pub use ffi::*;

use self::error::CublasResult;

#[derive(Debug)]
pub struct CublasHandle(pub *mut cublasContext);

pub fn create_handle() -> CublasResult<CublasHandle> {
    let mut handle: CublasHandle = CublasHandle(std::ptr::null_mut());
    unsafe { cublasCreate_v2(&mut handle.0) }.to_result()?;
    Ok(handle)
}

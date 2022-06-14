mod ffi;
mod error;

use std::{ptr::null_mut, ffi::CString};

pub use ffi::*;

use self::error::NvrtcResult;

pub struct NvrtcProgram(pub nvrtcProgram);

pub fn create_program(src: &str, name: &str) -> NvrtcResult<NvrtcProgram> {
    let src = CString::new(src).unwrap();
    let name = CString::new(name).unwrap();

    let mut prog = NvrtcProgram(null_mut());
    unsafe { nvrtcCreateProgram(&mut prog.0, src.as_ptr(), name.as_ptr(), 0, null_mut(), null_mut()) }.to_result()?;
    Ok(prog)
}
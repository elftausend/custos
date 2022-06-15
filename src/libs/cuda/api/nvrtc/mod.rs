mod ffi;
mod error;

use std::{ptr::null_mut, ffi::CString};

pub use ffi::*;

use self::error::NvrtcResult;

pub struct NvrtcProgram(pub nvrtcProgram);

impl NvrtcProgram {
    pub fn compile(&self) -> NvrtcResult<()> {
        compile_program(self)
    }
    
    pub fn ptx(&self) -> NvrtcResult<CString> {
        get_ptx(self)
    }
}

pub fn create_program(src: &str, name: &str) -> NvrtcResult<NvrtcProgram> {
    let src = CString::new(src).unwrap();
    let name = CString::new(name).unwrap();

    let mut prog = NvrtcProgram(null_mut());
    unsafe { nvrtcCreateProgram(&mut prog.0, src.as_ptr(), name.as_ptr(), 0, null_mut(), null_mut()) }.to_result()?;
    Ok(prog)
}
pub fn compile_program(prog: &NvrtcProgram) -> NvrtcResult<()> {
    unsafe { nvrtcCompileProgram(prog.0, 0, null_mut()) }.to_result()
} 

pub fn get_ptx(prog: &NvrtcProgram) -> NvrtcResult<CString> {
    unsafe {
        let mut ptx_size = 0;
        nvrtcGetPTXSize(prog.0, &mut ptx_size).to_result()?;
        let mut src: Vec<u8> = vec![0; ptx_size as usize];
        nvrtcGetPTX(prog.0, src.as_mut_ptr() as *mut i8).to_result()?;
        Ok(CString::from_vec_with_nul_unchecked(src))
    }

}
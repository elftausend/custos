mod error;
mod ffi;

use std::{
    ffi::CString,
    ptr::{null, null_mut}, os::raw::c_char,
};

pub use ffi::*;

use self::error::NvrtcResult;

pub struct NvrtcProgram(pub nvrtcProgram);

impl NvrtcProgram {
    pub fn compile(&self, options: Option<Vec<CString>>) -> NvrtcResult<()> {
        compile_program(self, options)
    }

    pub fn ptx(&self) -> NvrtcResult<CString> {
        get_ptx(self)
    }
}

pub fn create_program(src: &str, name: &str) -> NvrtcResult<NvrtcProgram> {
    let src = CString::new(src).unwrap();
    let name = CString::new(name).unwrap();

    let mut prog = NvrtcProgram(null_mut());
    unsafe {
        nvrtcCreateProgram(
            &mut prog.0,
            src.as_ptr(),
            name.as_ptr(),
            0,
            null_mut(),
            null_mut(),
        )
    }
    .to_result()?;
    Ok(prog)
}
pub fn compile_program(prog: &NvrtcProgram, options: Option<Vec<CString>>) -> NvrtcResult<()> {
    /*
    let (num_options, options) = match options {
        Some(options) => (options.len(), options.as_ptr()),
        None => (0, null()),
    };
    */
    match options {
        Some(options) => {
            let options = options
                .iter()
                .map(|option| option.as_ptr())
                .collect::<Vec<*const c_char>>();
            unsafe { nvrtcCompileProgram(prog.0, options.len() as i32, options.as_ptr()) }
                .to_result()
        }
        None => unsafe { nvrtcCompileProgram(prog.0, 0, null()) }.to_result(),
    }
    //unsafe { nvrtcCompileProgram(prog.0, num_options as i32, options as *const *const i8) }.to_result()
}

pub fn get_ptx(prog: &NvrtcProgram) -> NvrtcResult<CString> {
    unsafe {
        let mut ptx_size = 0;
        nvrtcGetPTXSize(prog.0, &mut ptx_size).to_result()?;
        let mut src: Vec<u8> = vec![0; ptx_size as usize];
        nvrtcGetPTX(prog.0, src.as_mut_ptr() as *mut c_char).to_result()?;
        Ok(CString::from_vec_with_nul_unchecked(src))
    }
}

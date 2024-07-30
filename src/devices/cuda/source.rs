use core::fmt::Display;
use std::ffi::CString;

use crate::cuda::api::nvrtc::create_program;
use crate::Error;

use super::api::nvrtc::nvrtcDestroyProgram;

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ptx {
    pub src: String,
}

impl Display for Ptx {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.src)
    }
}

pub trait CudaSource: Display {
    fn as_src_str(&self) -> String {
        self.to_string()
    }
    fn ptx(&self) -> Result<CString, Error>;
}

impl CudaSource for Ptx {
    #[inline]
    fn ptx(&self) -> Result<CString, Error> {
        Ok(CString::new(&*self.src).unwrap())
    }
}

fn compile_cuda_src_to_ptx(src: impl AsRef<str>) -> Result<CString, Error> {
    // TODO: not optimal, if multiple functions are used in the same source code, they are compiled multiple times
    let mut x = create_program(src.as_ref(), "")?;

    x.compile(Some(vec![c"--use_fast_math".to_owned()]))?;
    let ptx = x.ptx()?;

    unsafe { nvrtcDestroyProgram(&mut x.0).to_result()? };

    Ok(ptx)
}

impl CudaSource for String {
    #[inline]
    fn ptx(&self) -> Result<CString, Error> {
        compile_cuda_src_to_ptx(self)
    }
}

impl CudaSource for &String {
    #[inline]
    fn ptx(&self) -> Result<CString, Error> {
        compile_cuda_src_to_ptx(self)
    }
}

impl CudaSource for &str {
    #[inline]
    fn ptx(&self) -> Result<CString, Error> {
        compile_cuda_src_to_ptx(self)
    }
}

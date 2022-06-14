#![allow(dead_code)]
#![allow(non_camel_case_types)]

use super::error::{NvrtcErrorKind, NvrtcResult};

pub enum _nvrtcProgram {}
pub type nvrtcProgram = *mut _nvrtcProgram;

#[repr(C)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
}

impl nvrtcResult {
    pub fn to_result(self) -> NvrtcResult<()> {
        match self {
            nvrtcResult::NVRTC_SUCCESS => Ok(()),
            _ => Err(NvrtcErrorKind::from(self as u32))
        }
    }
}

#[link(name="nvrtc")]
extern "C" {
    pub fn nvrtcCreateProgram(prog: *mut nvrtcProgram, src: *const i8, name: *const i8, numHeaders: i32, headers: *const *const i8, includeNames: *const *const i8) -> nvrtcResult;
    pub fn nvrtcCompileProgram(prog: nvrtcProgram, numOptions: i32, options: *const *const i8) -> nvrtcResult;
    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut i8) -> nvrtcResult;
    pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptx_size: *mut isize) -> nvrtcResult;
}
//! NVRTC ffi modulue

mod error;
mod ffi;

use std::{
    ffi::CString,
    os::raw::c_char,
    ptr::{null, null_mut},
};

use self::error::NvrtcResult;
pub use ffi::*;

/// A compileable nvrtc program
pub struct NvrtcProgram(pub nvrtcProgram);

impl NvrtcProgram {
    /// Compiles the nvrtc program
    pub fn compile(&self, options: Option<Vec<CString>>) -> NvrtcResult<()> {
        compile_program(self, options)
    }

    /// Returns the runnable ptx
    pub fn ptx(&self) -> NvrtcResult<CString> {
        get_ptx(self)
    }
}

/// creates a new compileable nvrtc program
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

/// Compiles a nvrtc program
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

/// Returns runnable ptx
pub fn get_ptx(prog: &NvrtcProgram) -> NvrtcResult<CString> {
    unsafe {
        let mut ptx_size = 0;
        nvrtcGetPTXSize(prog.0, &mut ptx_size).to_result()?;
        let mut src: Vec<u8> = vec![0; ptx_size as usize];
        nvrtcGetPTX(prog.0, src.as_mut_ptr() as *mut c_char).to_result()?;
        Ok(CString::from_vec_with_nul_unchecked(src))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cuda::api::{culaunch_kernel, load_module_data, nvrtc::create_program},
        Base, Buffer, Read, CUDA,
    };
    use std::ffi::c_void;

    #[test]
    fn test_nvrtc() -> crate::Result<()> {
        let device = CUDA::<Base>::new(0)?;

        let a = Buffer::from((&device, [1, 2, 3, 4, 5]));
        let b = Buffer::from((&device, [4, 1, 7, 6, 9]));

        let c = Buffer::<i32, _>::new(&device, a.len());

        let src = r#"
            extern "C" __global__ void add(int *a, int *b, int *c, int numElements)
            {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {
                    c[idx] = a[idx] + b[idx];
                }
        }"#;

        let x = create_program(src, "add")?;
        x.compile(None)?;
        let module = load_module_data(x.ptx()?)?;
        let function = module.function("add")?;

        culaunch_kernel(
            &function,
            [a.len() as u32, 1, 1],
            [1, 1, 1],
            0,
            device.stream(),
            &mut [
                &a.ptrs().2 as *const u64 as *mut c_void,
                &b.ptrs().2 as *const u64 as *mut c_void,
                &c.ptrs().2 as *const u64 as *mut c_void,
                &a.len() as *const usize as *mut c_void,
            ],
        )?;

        let read = device.read(&c);
        assert_eq!(vec![5, 3, 10, 10, 14], read);
        Ok(())
    }
}

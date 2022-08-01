use super::api::{
    build_program, create_kernels_in_program, create_program_with_source, release_mem_object, Kernel
};
use crate::{CLDevice, Error, libs::cache::CacheType};
use std::{collections::HashMap, ffi::c_void};


#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OclPtr(pub *mut c_void);

unsafe impl Send for OclPtr {}
unsafe impl Sync for OclPtr {}

#[derive(Debug)]
pub struct RawCL {
    pub ptr: *mut c_void,
    pub host_ptr: *mut u8,
}

impl CacheType for RawCL {
    fn new<T>(ptr: (*mut T, *mut c_void, u64), _: usize) -> Self {
        RawCL {
            ptr: ptr.1,
            host_ptr: ptr.0 as *mut u8,
        }
    }

    fn destruct<T>(&self) -> (*mut T, *mut c_void, u64) {
        (self.host_ptr as *mut T, self.ptr, 0)
    }
}

impl Drop for RawCL {
    fn drop(&mut self) {
        unsafe { 
            release_mem_object(self.ptr).unwrap() 
        };
    }
}

#[derive(Debug)]
/// Stores kernels and outputs
pub struct KernelCache {
    pub(crate) kernel_cache: HashMap<String, Kernel>,
}

impl KernelCache {
    pub fn new() -> KernelCache {
        KernelCache { kernel_cache: HashMap::new() }
    }
    pub fn kernel_cache(&mut self, device: &CLDevice, src: &str) -> Result<Kernel, Error> {
        let kernel = self.kernel_cache.get(src);

        if let Some(kernel) = kernel {
            return Ok(*kernel);
        }

        let program = create_program_with_source(&device.ctx(), src)?;
        build_program(&program, &[device.device()], Some("-cl-std=CL1.2"))?; //-cl-single-precision-constant
        let kernel = create_kernels_in_program(&program)?[0];

        self.kernel_cache.insert(src.to_string(), kernel);
        Ok(kernel)
    }
}

impl Drop for KernelCache {
    fn drop(&mut self) {
        // FIXME:
        // TODO:  not really safe
        for kernel in &mut self.kernel_cache.values_mut() {
            kernel.release()
        }    
    }
}

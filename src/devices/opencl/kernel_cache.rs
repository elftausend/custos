use super::api::{
    build_program, create_kernels_in_program, create_program_with_source, release_mem_object,
    Kernel,
};
use crate::{devices::cache::CacheType, CLDevice, Error, Node};
use std::{collections::HashMap, ffi::c_void};

#[derive(Debug)]
pub struct RawCL {
    pub ptr: *mut c_void,
    pub host_ptr: *mut u8,
    pub node: Node,
}

impl CacheType for RawCL {
    fn new<T>(ptr: (*mut T, *mut c_void, u64), _: usize, node: Node) -> Self {
        RawCL {
            ptr: ptr.1,
            host_ptr: ptr.0 as *mut u8,
            node,
        }
    }

    fn destruct<T>(&self) -> ((*mut T, *mut c_void, u64), Node) {
        ((self.host_ptr as *mut T, self.ptr, 0), self.node)
    }
}

impl Drop for RawCL {
    fn drop(&mut self) {
        unsafe { release_mem_object(self.ptr).unwrap() };
    }
}

#[derive(Debug, Default)]
/// This stores the previously compiled OpenCL kernels.
pub struct KernelCacheCL {
    pub(crate) kernel_cache: HashMap<String, Kernel>,
}

impl KernelCacheCL {
    pub fn kernel_cache(&mut self, device: &CLDevice, src: &str) -> Result<Kernel, Error> {
        if let Some(kernel) = self.kernel_cache.get(src) {
            return Ok(*kernel);
        }

        let program = create_program_with_source(&device.ctx(), src)?;
        build_program(&program, &[device.device()], Some("-cl-std=CL1.2"))?; //-cl-single-precision-constant
        let kernel = create_kernels_in_program(&program)?[0];

        self.kernel_cache.insert(src.to_string(), kernel);
        Ok(kernel)
    }
}

impl Drop for KernelCacheCL {
    fn drop(&mut self) {
        // FIXME:
        // TODO:  not really safe
        for kernel in &mut self.kernel_cache.values_mut() {
            kernel.release()
        }
    }
}

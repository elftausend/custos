use super::api::{
    load_module_data,
    nvrtc::{create_program, nvrtcDestroyProgram},
    FnHandle,
};
use crate::{Error, CUDA};
use std::{collections::HashMap, ffi::CString};

/// This stores the previously compiled CUDA functions / kernels.
#[derive(Debug, Default)]
pub struct KernelCacheCU {
    /// Uses the kernel source code to retrieve the corresponding `FnHandle`.
    pub kernels: HashMap<String, FnHandle>,
}

impl KernelCacheCU {
    /// Returns a cached kernel. If the kernel source code does not exist, a new kernel is created and cached.
    ///
    /// # Example
    /// ```
    /// use std::collections::HashMap;
    /// use custos::{CUDA, cuda::KernelCacheCU};
    ///
    /// fn main() -> custos::Result<()> {
    ///     let device = CUDA::new(0)?;
    ///     
    ///     let mut kernel_cache = KernelCacheCU::default();
    ///     
    ///     let mut kernel_fn = || kernel_cache.kernel(&device, r#"
    ///         extern "C" __global__ void test(float* test) {}
    ///     "#, "test").unwrap().0;
    ///     
    ///     let kernel = kernel_fn();
    ///     let same_kernel = kernel_fn();
    ///     
    ///     assert_eq!(kernel, same_kernel);
    ///     Ok(())
    /// }
    /// ```
    pub fn kernel(&mut self, device: &CUDA, src: &str, fn_name: &str) -> Result<FnHandle, Error> {
        let kernel = self.kernels.get(src);

        if let Some(kernel) = kernel {
            return Ok(*kernel);
        }

        let mut x = create_program(src, "")?;

        x.compile(Some(vec![CString::new("--use_fast_math").unwrap()]))?;

        let module = load_module_data(x.ptx()?)?;
        let function = module.function(fn_name)?;

        device.modules.borrow_mut().push(module);

        self.kernels.insert(src.into(), function);
        unsafe { nvrtcDestroyProgram(&mut x.0).to_result()? };
        Ok(function)
    }
}

/// Exactly like [`KernelCacheCU`], but with a immutable source of the cache using interior mutability.
pub fn fn_cache(device: &CUDA, src: &str, fn_name: &str) -> crate::Result<FnHandle> {
    device
        .kernel_cache
        .borrow_mut()
        .kernel(device, src, fn_name)
}

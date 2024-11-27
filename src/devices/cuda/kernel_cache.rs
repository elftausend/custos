use super::{
    CudaDevice, CudaSource,
    api::{FnHandle, Module, load_module_data},
};
use crate::Error;
use std::collections::HashMap;

/// This stores the previously compiled CUDA functions / kernels.
#[derive(Debug, Default)]
pub struct KernelCache {
    /// Uses the kernel source code and the kernel function to retrieve the corresponding `FnHandle`.
    pub kernels: HashMap<(String, String), FnHandle>,
}

impl KernelCache {
    /// Returns a cached kernel. If the kernel source code does not exist, a new kernel is created and cached.
    ///
    /// # Example
    /// ```
    /// use std::collections::HashMap;
    /// use custos::{CUDA, cuda::KernelCache, Base};
    ///
    /// fn main() -> custos::Result<()> {
    ///     let device = CUDA::<Base>::new(0)?;
    ///     
    ///     let mut kernel_cache = KernelCache::default();
    ///     
    ///     let mut kernel_fn = || kernel_cache.kernel(&mut device.cuda_modules.borrow_mut(), r#"
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
    pub fn kernel(
        &mut self,
        modules: &mut HashMap<FnHandle, Module>,
        src: impl CudaSource,
        fn_name: &str,
    ) -> Result<FnHandle, Error> {
        let kernel = self.kernels.get(&(src.as_src_str(), fn_name.into()));

        if let Some(kernel) = kernel {
            return Ok(*kernel);
        }

        let module = load_module_data(src.ptx()?)?;
        let function = module.function(fn_name)?;

        // TODO: not optimal, if multiple functions are used in the same source code, they are compiled multiple times
        modules.insert(function, module);

        self.kernels
            .insert((src.as_src_str(), fn_name.into()), function);
        Ok(function)
    }
}

/// Exactly like [`KernelCacheCU`], but with a immutable source of the cache using interior mutability.
pub fn fn_cache(
    device: &CudaDevice,
    src: impl CudaSource,
    fn_name: &str,
) -> crate::Result<FnHandle> {
    device
        .kernel_cache
        .borrow_mut()
        .kernel(&mut device.cuda_modules.borrow_mut(), src, fn_name)
}

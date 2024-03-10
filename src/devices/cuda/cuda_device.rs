use core::cell::RefCell;
use std::collections::HashMap;

use super::{
    api::{
        create_context, create_stream, cuInit, cuStreamDestroy,
        cublas::{create_handle, cublasDestroy_v2, cublasSetStream_v2, CublasHandle},
        device, Context, CudaErrorKind, CudaIntDevice, FnHandle, Module, Stream,
    },
    AsCudaCvoidPtr, CudaSource, KernelCache,
};

pub struct CudaDevice {
    /// Stores compiled CUDA kernels.
    pub kernel_cache: RefCell<KernelCache>,
    /// Stores CUDA modules from the compiled kernels.
    pub cuda_modules: RefCell<HashMap<FnHandle, Module>>,
    device: CudaIntDevice,
    ctx: Context,
    /// The default stream used for operations.
    pub stream: Stream,
    /// A stream used for memory transfers, like cu_write_async
    pub mem_transfer_stream: Stream,
    pub handle: CublasHandle,
    #[cfg(feature = "lazy")]
    // TODO: remove result when get_or_try_init becomes stable
    pub graph: core::cell::OnceCell<Result<super::lazy::LazyCudaGraph, CudaErrorKind>>,
}

impl CudaDevice {
    pub fn new(idx: usize) -> crate::Result<Self> {
        unsafe { cuInit(0) }.to_result()?;
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let stream = create_stream()?;
        let mem_transfer_stream = create_stream()?;
        let handle = create_handle()?;
        unsafe { cublasSetStream_v2(handle.0, stream.0) }.to_result()?;

        Ok(Self {
            kernel_cache: Default::default(),
            cuda_modules: Default::default(),
            device,
            ctx,
            stream,
            mem_transfer_stream,
            handle,
            #[cfg(feature = "lazy")]
            graph: core::cell::OnceCell::new(),
        })
    }
}

impl CudaDevice {
    /// Returns the internal CUDA device.
    #[inline]
    pub fn device(&self) -> &CudaIntDevice {
        &self.device
    }

    /// Returns the internal CUDA context.
    #[inline]
    pub fn ctx(&self) -> &Context {
        &self.ctx
    }

    /// Returns the cublas handle
    #[inline]
    pub fn cublas_handle(&self) -> &CublasHandle {
        &self.handle
    }

    /// Returns the internal CUDA stream.
    #[inline]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Lauches a CUDA kernel with the given arguments.
    #[inline]
    pub fn launch_kernel1d(
        &self,
        len: usize,
        src: impl CudaSource,
        fn_name: &str,
        args: &[&dyn AsCudaCvoidPtr],
    ) -> crate::Result<()> {
        super::launch_kernel1d(
            len,
            &mut self.kernel_cache.borrow_mut(),
            &mut self.cuda_modules.borrow_mut(),
            self.stream(),
            src,
            fn_name,
            args,
        )
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // deallocates all cached buffers before destroying the context etc
        // TODO: keep in mind
        // self.cache_mut().nodes.clear();

        unsafe {
            cublasDestroy_v2(self.handle.0);
            cuStreamDestroy(self.stream.0);

            cuStreamDestroy(self.mem_transfer_stream.0);
        }
    }
}

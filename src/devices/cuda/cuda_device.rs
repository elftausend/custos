use std::marker::PhantomData;
use std::{cell::RefCell, collections::HashMap};

use super::api::FnHandle;
use super::{
    api::{
        create_context, create_stream, cuInit, cuMemcpy, cuStreamDestroy, cu_write,
        cublas::{create_handle, cublasDestroy_v2, cublasSetStream_v2, CublasHandle},
        cumalloc, device, Context, CudaIntDevice, Module, Stream,
    },
    chosen_cu_idx, launch_kernel1d, AsCudaCvoidPtr, CUDAPtr, CUKernelCache,
};

use crate::{
    cache::Cache, flag::AllocFlag, Addons, AddonsReturn, Alloc, Buffer, CacheReturn, CloneBuf,
    Device, PtrConv, Shape,
};

/// Used to perform calculations with a CUDA capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [`CUDA`].
#[derive(Debug)]
pub struct CUDA {
    /// Stores compiled CUDA kernels.
    pub kernel_cache: RefCell<CUKernelCache>,
    /// Stores CUDA modules from the compiled kernels.
    pub modules: RefCell<HashMap<FnHandle, Module>>,
    device: CudaIntDevice,
    ctx: Context,
    stream: Stream,
    handle: CublasHandle,
    /// Provides additional functionality for the CUDA device. e.g. a cache, a gradient [`Tape`](crate::Tape), an optimizeable [`Graph`](crate::Graph) and a [`Cache`](crate::Cache).
    pub addons: Addons<CUDA>,
}

/// Short form for `CUDA`
pub type CU = CUDA;

impl CUDA {
    /// Returns an [CUDA] device at the specified device index.
    /// # Errors
    /// - No device was found at the given device index
    /// - some other CUDA related errors
    pub fn new(idx: usize) -> crate::Result<CUDA> {
        unsafe { cuInit(0) }.to_result()?;
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let stream = create_stream()?;
        let handle = create_handle()?;
        unsafe { cublasSetStream_v2(handle.0, stream.0) }.to_result()?;

        Ok(CUDA {
            kernel_cache: Default::default(),
            modules: Default::default(),
            addons: Default::default(),
            device,
            ctx,
            stream,
            handle,
        })
    }

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
        src: &str,
        fn_name: &str,
        args: &[&dyn AsCudaCvoidPtr],
    ) -> crate::Result<()> {
        launch_kernel1d(
            len,
            &mut self.kernel_cache.borrow_mut(),
            &mut self.modules.borrow_mut(),
            self.stream(),
            src,
            fn_name,
            args,
        )
    }
}

impl Device for CUDA {
    type Ptr<U, S: Shape> = CUDAPtr<U>;
    type Cache = Cache<CUDA>;

    fn new() -> crate::Result<Self> {
        CUDA::new(chosen_cu_idx())
    }
}

impl AddonsReturn for CUDA {
    #[inline]
    fn addons(&self) -> &Addons<Self>
    where
        Self: Device,
    {
        &self.addons
    }
}

impl PtrConv for CUDA {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Ptr<T, IS>,
        flag: AllocFlag,
    ) -> Self::Ptr<Conv, OS> {
        CUDAPtr {
            ptr: ptr.ptr,
            len: ptr.len,
            flag,
            p: PhantomData,
        }
    }
}

impl Default for CUDA {
    #[inline]
    fn default() -> Self {
        CUDA::new(chosen_cu_idx()).expect("A valid CUDA device index should be set via the environment variable `CUSTOS_CL_DEVICE_IDX`")
    }
}

impl Drop for CUDA {
    fn drop(&mut self) {
        // deallocates all cached buffers before destroying the context etc
        self.cache_mut().nodes.clear();

        unsafe {
            cublasDestroy_v2(self.handle.0);
            cuStreamDestroy(self.stream.0);
        }
    }
}

impl<T> Alloc<'_, T> for CUDA {
    fn alloc(&self, len: usize, flag: AllocFlag) -> CUDAPtr<T> {
        let ptr = cumalloc::<T>(len).unwrap();
        // TODO: use unified mem if available -> i can't test this
        CUDAPtr {
            ptr,
            len,
            flag,
            p: PhantomData,
        }
    }

    fn with_slice(&self, data: &[T]) -> CUDAPtr<T> {
        let ptr = cumalloc::<T>(data.len()).unwrap();
        cu_write(ptr, data).unwrap();
        CUDAPtr {
            ptr,
            len: data.len(),
            flag: AllocFlag::None,
            p: PhantomData,
        }
    }
}

impl<'a, T> CloneBuf<'a, T> for CUDA {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CUDA>) -> Buffer<'a, T, CUDA> {
        let cloned = Buffer::new(self, buf.len());
        unsafe {
            cuMemcpy(
                cloned.ptrs().2,
                buf.ptrs().2,
                buf.len() * std::mem::size_of::<T>(),
            );
        }
        cloned
    }
}

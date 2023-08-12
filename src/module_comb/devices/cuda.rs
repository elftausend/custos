use core::{cell::RefCell, marker::PhantomData};
use std::collections::HashMap;

use crate::{
    cuda::{
        api::{
            create_context, create_stream, cuInit, cuStreamDestroy,
            cublas::{create_handle, cublasDestroy_v2, cublasSetStream_v2, CublasHandle},
            cumalloc, device, Context, CudaIntDevice, FnHandle, Module, Stream,
        },
        launch_kernel1d, AsCudaCvoidPtr, CUDAPtr, CUKernelCache, CudaSource,
    },
    impl_buffer_hook_traits, impl_retriever,
    module_comb::{
        Alloc, Buffer, LazySetup, OnDropBuffer, OnNewBuffer, Retrieve, Retriever, Setup,
    },
    Shape,
};

use super::Device;

pub trait IsCuda: Device {}

pub struct CUDA<Mods> {
    modules: Mods,
    /// Stores compiled CUDA kernels.
    pub kernel_cache: RefCell<CUKernelCache>,
    /// Stores CUDA modules from the compiled kernels.
    pub cuda_modules: RefCell<HashMap<FnHandle, Module>>,
    device: CudaIntDevice,
    ctx: Context,
    stream: Stream,
    handle: CublasHandle,
}

impl_retriever!(CUDA);
impl_buffer_hook_traits!(CUDA);

impl<SimpleMods> CUDA<SimpleMods> {
    /// Returns an [CUDA] device at the specified device index.
    /// # Errors
    /// - No device was found at the given device index
    /// - some other CUDA related errors
    #[inline]
    pub fn new<NewMods>(idx: usize) -> crate::Result<CUDA<NewMods>>
    where
        SimpleMods: crate::module_comb::Module<CUDA<SimpleMods>, Module = NewMods>,
        NewMods: Setup<CUDA<NewMods>>,
    {
        unsafe { cuInit(0) }.to_result()?;
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let stream = create_stream()?;
        let handle = create_handle()?;
        unsafe { cublasSetStream_v2(handle.0, stream.0) }.to_result()?;

        let mut cuda = CUDA {
            modules: SimpleMods::new(),
            kernel_cache: Default::default(),
            cuda_modules: Default::default(),
            device,
            ctx,
            stream,
            handle,
        };

        NewMods::setup(&mut cuda);

        Ok(cuda)
    }
}

impl<Mods> CUDA<Mods> {
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
        launch_kernel1d(
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

impl<Mods: OnDropBuffer> Device for CUDA<Mods> {
    type Error = i32;
}

impl<Mods> Drop for CUDA<Mods> {
    fn drop(&mut self) {
        // deallocates all cached buffers before destroying the context etc
        // TODO: keep in mind
        // self.cache_mut().nodes.clear();

        unsafe {
            cublasDestroy_v2(self.handle.0);
            cuStreamDestroy(self.stream.0);
        }
    }
}

impl<Mods> Alloc for CUDA<Mods> {
    type Data<T, S: Shape> = CUDAPtr<T>;

    fn alloc<T, S: Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Data<T, S> {
        let ptr = cumalloc::<T>(len).unwrap();
        // TODO: use unified mem if available -> i can't test this
        CUDAPtr {
            ptr,
            len,
            flag,
            p: PhantomData,
        }
    }

    fn alloc_from_slice<T, S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone,
    {
        todo!()
    }
}

impl<Mods: OnDropBuffer> IsCuda for CUDA<Mods> {}

impl<Mods> LazySetup for CUDA<Mods> {
    #[inline]
    fn lazy_setup(&mut self) {
        // switch to stream record mode for graph
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        module_comb::{Base, Buffer, Retriever},
        Shape,
    };

    use super::{IsCuda, CUDA};

    fn take_cu_buffer<T: 'static, D: IsCuda + Retriever, S: Shape>(
        device: &D,
        buf: &Buffer<T, D, S>,
    ) {
        let buf = device.retrieve::<T, S, 0>(buf.len(), ());
    }

    #[test]
    fn test_cu_buffer_fn() {
        let device = CUDA::<Base>::new(0).unwrap();
        let buf = Buffer::<f32, _, ()>::new(&device, 10);
        take_cu_buffer(&device, &buf)
    }
}

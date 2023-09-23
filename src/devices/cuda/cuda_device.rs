use core::{
    cell::{OnceCell, RefCell},
    marker::PhantomData,
};
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
    flag::AllocFlag,
    impl_retriever, Alloc, Base, BufAsLcd, Buffer, CloneBuf, Device, Module as CombModule, PtrConv,
    Retrieve, Setup, Shape,
};

use super::api::{cuMemcpy, cu_write_async};

pub trait IsCuda: Device {}

/// Used to perform calculations with a CUDA capable device.
pub struct CUDA<Mods = Base> {
    modules: Mods,
    /// Stores compiled CUDA kernels.
    pub kernel_cache: RefCell<CUKernelCache>,
    /// Stores CUDA modules from the compiled kernels.
    pub cuda_modules: RefCell<HashMap<FnHandle, Module>>,
    device: CudaIntDevice,
    ctx: Context,
    /// The default stream used for operations.
    pub stream: Stream,
    /// A stream used for memory transfers, like cu_write_async
    pub mem_transfer_stream: Stream,
    handle: CublasHandle,
    #[cfg(feature = "lazy")]
    pub graph: OnceCell<super::lazy::LazyCudaGraph>,
}

impl_retriever!(CUDA);

impl<SimpleMods> CUDA<SimpleMods> {
    /// Returns an [CUDA] device at the specified device index.
    /// # Errors
    /// - No device was found at the given device index
    /// - some other CUDA related errors
    #[inline]
    pub fn new<NewMods>(idx: usize) -> crate::Result<CUDA<NewMods>>
    where
        SimpleMods: CombModule<CUDA, Module = NewMods>,
        NewMods: Setup<CUDA<NewMods>>,
    {
        unsafe { cuInit(0) }.to_result()?;
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let stream = create_stream()?;
        let mem_transfer_stream = create_stream()?;
        let handle = create_handle()?;
        unsafe { cublasSetStream_v2(handle.0, stream.0) }.to_result()?;

        let mut cuda = CUDA {
            modules: SimpleMods::new(),
            kernel_cache: Default::default(),
            cuda_modules: Default::default(),
            device,
            ctx,
            stream,
            mem_transfer_stream,
            handle,
            #[cfg(feature = "lazy")]
            graph: OnceCell::new(),
        };

        NewMods::setup(&mut cuda)?;

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
    pub fn launch_kernel1d<'a>(
        &self,
        len: usize,
        src: impl CudaSource,
        fn_name: &str,
        args: &'a [&dyn AsCudaCvoidPtr],
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

impl<Mods> Device for CUDA<Mods> {
    type Data<T, S: Shape> = CUDAPtr<T>;
    type Error = i32;
}

impl<Mods> BufAsLcd for CUDA<Mods> {
    type LCD<T> = CUDAPtr<T>;

    #[inline]
    fn lcd<'a, 'b, T, S: Shape>(&'a self, buf: &'b Buffer<'a, T, Self, S>) -> &'b Self::LCD<T> {
        &buf.data
    }

    #[inline]
    fn lcd_mut<'a, 'b, T, S: Shape>(
        &'a self,
        buf: &'b mut Buffer<'a, T, Self, S>,
    ) -> &'b mut Self::LCD<T> {
        &mut buf.data
    }
}

impl<Mods> Drop for CUDA<Mods> {
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

impl<Mods, T> Alloc<T> for CUDA<Mods> {
    fn alloc<S: Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Data<T, S> {
        let ptr = cumalloc::<T>(len).unwrap();
        // TODO: use unified mem if available -> i can't test this
        CUDAPtr {
            ptr,
            len,
            flag,
            p: PhantomData,
        }
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T], alloc_flag: AllocFlag) -> Self::Data<T, S>
    where
        T: Clone,
    {
        let ptr = cumalloc::<T>(data.len()).unwrap();
        cu_write_async(ptr, data, &self.mem_transfer_stream).unwrap();
        self.mem_transfer_stream.sync().unwrap();
        CUDAPtr {
            ptr,
            len: data.len(),
            flag: alloc_flag,
            p: PhantomData,
        }
    }
}

impl<Mods> IsCuda for CUDA<Mods> {}

#[cfg(feature = "fork")]
impl<Mods> crate::ForkSetup for CUDA<Mods> {
    #[inline]
    fn fork_setup(&mut self) {
        // TODO: maybe check if device supports unified memory
    }
}

impl<Mods> PtrConv for CUDA<Mods> {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Data<T, IS>,
        flag: AllocFlag,
    ) -> Self::Data<Conv, OS> {
        CUDAPtr {
            ptr: ptr.ptr,
            len: ptr.len,
            flag,
            p: PhantomData,
        }
    }
}

impl<'a, Mods: Retrieve<Self, T, ()>, T> CloneBuf<'a, T> for CUDA<Mods> {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CUDA<Mods>>) -> Buffer<'a, T, CUDA<Mods>> {
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

#[cfg(test)]
mod tests {
    use crate::{Base, Buffer, Retriever, Shape};

    use super::{IsCuda, CUDA};

    // compile-time isCuda test
    fn take_cu_buffer<T, D: IsCuda + Retriever<T, S>, S: Shape>(device: &D, buf: &Buffer<T, D, S>) {
        let _buf = device.retrieve::<0>(buf.len(), ());
    }

    #[test]
    fn test_cu_buffer_fn() {
        let device = CUDA::<Base>::new(0).unwrap();
        let buf = Buffer::<f32, _, ()>::new(&device, 10);
        take_cu_buffer(&device, &buf)
    }
}

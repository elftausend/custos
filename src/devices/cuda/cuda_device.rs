use std::cell::RefCell;
use std::marker::PhantomData;

use super::{
    api::{
        create_context, create_stream, cuInit, cuMemcpy, cuStreamDestroy, cu_write,
        cublas::{create_handle, cublasDestroy_v2, cublasSetStream_v2, CublasHandle},
        cumalloc, device, Context, CudaIntDevice, Module, Stream,
    },
    chosen_cu_idx, launch_kernel1d, AsCudaCvoidPtr, CUDAPtr, KernelCacheCU, RawCUBuf,
};

use crate::{
    cache::{Cache, CacheReturn},
    flag::AllocFlag,
    Alloc, Buffer, CloneBuf, Device, GlobalCount, Graph, GraphReturn, RawConv, Shape,
};

/// Used to perform calculations with a CUDA capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CudaDevice].
#[derive(Debug)]
pub struct CUDA {
    pub cache: RefCell<Cache<CUDA>>,
    pub kernel_cache: RefCell<KernelCacheCU>,
    pub modules: RefCell<Vec<Module>>,
    pub graph: RefCell<Graph<GlobalCount>>,
    device: CudaIntDevice,
    ctx: Context,
    stream: Stream,
    handle: CublasHandle,
}

/// Short form for `CUDA`
pub type CU = CUDA;

impl CUDA {
    pub fn new(idx: usize) -> crate::Result<CUDA> {
        unsafe { cuInit(0) }.to_result()?;
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let stream = create_stream()?;
        let handle = create_handle()?;
        unsafe { cublasSetStream_v2(handle.0, stream.0) }.to_result()?;

        Ok(CUDA {
            cache: RefCell::new(Cache::default()),
            kernel_cache: RefCell::new(KernelCacheCU::default()),
            modules: RefCell::new(vec![]),
            graph: RefCell::new(Graph::new()),
            device,
            ctx,
            stream,
            handle,
        })
    }

    pub fn device(&self) -> &CudaIntDevice {
        &self.device
    }

    pub fn ctx(&self) -> &Context {
        &self.ctx
    }

    pub fn handle(&self) -> &CublasHandle {
        &self.handle
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    #[inline]
    pub fn launch_kernel1d(
        &self,
        len: usize,
        src: &str,
        fn_name: &str,
        args: &[&dyn AsCudaCvoidPtr],
    ) -> crate::Result<()> {
        launch_kernel1d(len, self, src, fn_name, args)
    }
}

impl Device for CUDA {
    type Ptr<U, S: Shape> = CUDAPtr<U>;
    type Cache = Cache<CUDA>;

    fn new() -> crate::Result<Self> {
        CUDA::new(chosen_cu_idx())
    }
}

impl RawConv for CUDA {
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, flag: AllocFlag) -> Self::CT {
        RawCUBuf {
            ptr: ptr.ptr,
            flag,
            len,
        }
    }

    fn destruct<T, S: Shape>(ct: &Self::CT) -> Self::Ptr<T, S> {
        CUDAPtr {
            ptr: ct.ptr,
            len: ct.len,
            flag: ct.flag,
            p: PhantomData,
        }
    }
}

impl Drop for CUDA {
    fn drop(&mut self) {
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

impl GraphReturn for CUDA {
    fn graph(&self) -> std::cell::Ref<Graph<GlobalCount>> {
        self.graph.borrow()
    }

    fn graph_mut(&self) -> std::cell::RefMut<Graph<GlobalCount>> {
        self.graph.borrow_mut()
    }
}

impl CacheReturn for CUDA {
    type CT = RawCUBuf;
    #[inline]
    fn cache(&self) -> std::cell::RefMut<Cache<CUDA>> {
        self.cache.borrow_mut()
    }
}

#[cfg(feature = "opt-cache")]
impl crate::GraphOpt for CUDA {}

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

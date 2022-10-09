use super::{
    api::{
        create_context, create_stream, cuInit, cuMemcpy, cuStreamDestroy, cu_read, cu_write,
        cublas::{create_handle, cublasDestroy_v2, cublasSetStream_v2, CublasHandle},
        cumalloc, device, Context, CudaIntDevice, Module, Stream,
    },
    cu_clear, CUDAPtr, KernelCacheCU, RawCUBuf,
};
use crate::{
    cache::{Cache, CacheReturn},
    Alloc, Buffer, CDatatype, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, Device, Graph, GraphReturn,
    VecRead, WriteBuf,
};
use std::{cell::RefCell, marker::PhantomData};

/// Used to perform calculations with a CUDA capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CudaDevice].
#[derive(Debug)]
pub struct CUDA {
    pub cache: RefCell<Cache<RawCUBuf>>,
    pub kernel_cache: RefCell<KernelCacheCU>,
    pub modules: RefCell<Vec<Module>>,
    pub graph: RefCell<Graph>,
    device: CudaIntDevice,
    ctx: Context,
    stream: Stream,
    handle: CublasHandle,
}

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
}

impl Device for CUDA {
    type Ptr<U, const N: usize> = CUDAPtr<U>;
    type Cache<const N: usize> = Cache<RawCUBuf>;
}

impl Drop for CUDA {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.handle.0);
            cuStreamDestroy(self.stream.0);
        }
    }
}

impl<T> Alloc<T> for CUDA {
    fn alloc(&self, len: usize) -> CUDAPtr<T> {
        let ptr = cumalloc::<T>(len).unwrap();
        // TODO: use unified mem if available -> i can't test this
        CUDAPtr {
            ptr,
            p: PhantomData,
        }
    }

    fn with_slice(&self, data: &[T]) -> CUDAPtr<T> {
        let ptr = cumalloc::<T>(data.len()).unwrap();
        cu_write(ptr, data).unwrap();
        CUDAPtr {
            ptr,
            p: PhantomData,
        }
    }
}

impl<T: Default + Clone> VecRead<T, CUDA> for CUDA {
    fn read(&self, buf: &Buffer<T, CUDA>) -> Vec<T> {
        assert!(
            buf.ptrs().2 != 0,
            "called VecRead::read(..) on a non CUDA buffer"
        );
        let mut read = vec![T::default(); buf.len];
        cu_read(&mut read, buf.ptrs().2).unwrap();
        read
    }
}

impl<T: CDatatype> ClearBuf<T, CUDA> for CUDA {
    fn clear(&self, buf: &mut Buffer<T, CUDA>) {
        cu_clear(self, buf).unwrap()
    }
}

impl<T> WriteBuf<T, CUDA> for CUDA {
    fn write(&self, buf: &mut Buffer<T, CUDA>, data: &[T]) {
        cu_write(buf.cu_ptr(), data).unwrap();
    }
}

impl GraphReturn for CUDA {
    fn graph(&self) -> std::cell::RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

impl CacheReturn for CUDA {
    type CT = RawCUBuf;
    #[inline]
    fn cache(&self) -> std::cell::RefMut<Cache<RawCUBuf>> {
        self.cache.borrow_mut()
    }
}

#[cfg(feature = "opt-cache")]
impl crate::GraphOpt for CUDA {}

impl<'a, T> CloneBuf<'a, T> for CUDA {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CUDA>) -> Buffer<'a, T, CUDA> {
        let cloned = Buffer::new(self, buf.len);
        unsafe {
            cuMemcpy(
                cloned.ptrs().2,
                buf.ptrs().2,
                buf.len * std::mem::size_of::<T>(),
            );
        }
        cloned
    }
}

impl<'a, T> CacheBuf<'a, T> for CUDA {
    #[inline]
    fn cached(&self, len: usize) -> Buffer<T, CUDA> {
        Cache::get(self, len, CachedLeaf)
    }
}

#[inline]
pub fn cu_cached<T>(device: &CUDA, len: usize) -> Buffer<T, CUDA> {
    device.cached(len)
}

use core::ops::RangeBounds;

use std::cell::RefCell;
use std::marker::PhantomData;

use super::{
    api::{
        create_context, create_stream, cuInit, cuMemcpy, cuStreamDestroy, cu_read, cu_write,
        cublas::{create_handle, cublasDestroy_v2, cublasSetStream_v2, CublasHandle},
        cumalloc, device, Context, CudaIntDevice, Module, Stream,
    },
    chosen_cu_idx, cu_clear, CUDAPtr, KernelCacheCU, RawCUBuf,
};
use crate::devices::bounds_to_range;
use crate::{
    cache::{Cache, CacheReturn},
    flag::AllocFlag,
    Alloc, Buffer, CDatatype, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, CopySlice, Device, Graph, GraphReturn,
    RawConv, Read, Shape, WriteBuf,
};

/// Used to perform calculations with a CUDA capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CudaDevice].
#[derive(Debug)]
pub struct CUDA {
    pub cache: RefCell<Cache<CUDA>>,
    pub kernel_cache: RefCell<KernelCacheCU>,
    pub modules: RefCell<Vec<Module>>,
    pub graph: RefCell<Graph>,
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
}

impl Device for CUDA {
    type Ptr<U, S: Shape> = CUDAPtr<U>;
    type Cache = Cache<CUDA>;

    fn new() -> crate::Result<Self> {
        CUDA::new(chosen_cu_idx())
    }
}

impl RawConv for CUDA {
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, node: crate::Node) -> Self::CT {
        RawCUBuf {
            ptr: ptr.ptr,
            node,
            len,
        }
    }

    fn destruct<T, S: Shape>(ct: &Self::CT, flag: AllocFlag) -> (Self::Ptr<T, S>, crate::Node) {
        (
            CUDAPtr {
                ptr: ct.ptr,
                len: ct.len,
                flag,
                p: PhantomData,
            },
            ct.node,
        )
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

impl<T: Default + Clone> Read<T, CUDA> for CUDA {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        CUDA: 'a;

    #[inline]
    fn read(&self, buf: &Buffer<T, CUDA>) -> Vec<T> {
        self.read_to_vec(buf)
    }

    fn read_to_vec(&self, buf: &Buffer<T, CUDA>) -> Vec<T>
    where
        T: Default + Clone,
    {
        assert!(
            buf.ptrs().2 != 0,
            "called Read::read(..) on a non CUDA buffer"
        );
        // TODO: sync here or somewhere else?
        self.stream.sync().unwrap();

        let mut read = vec![T::default(); buf.len()];
        cu_read(&mut read, buf.ptrs().2).unwrap();
        read
    }
}

impl<T: CDatatype> ClearBuf<T, CUDA> for CUDA {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, CUDA>) {
        cu_clear(self, buf).unwrap()
    }
}

impl<T> CopySlice<T> for CUDA {
    fn copy_slice<R: RangeBounds<usize>>(&self, buf: &Buffer<T, CUDA>, range: R) -> Buffer<T, Self> {
        let range = bounds_to_range(range, buf.len());
        let copied = Buffer::new(self, range.end - range.start);

        unsafe {
            cuMemcpy(
                copied.ptr.ptr,
                buf.ptr.ptr + (range.start * std::mem::size_of::<T>()) as u64,
                copied.len() * std::mem::size_of::<T>(),
            );
        }

        copied
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

use std::{cell::{RefCell, Ref, RefMut}, rc::Rc, ptr::null_mut};
use crate::{Device, ClearBuf, VecRead, CacheBuf, BaseDevice, GenericBlas, CDatatype, CUdeviceptr, AsDev};
use super::{api::{device, create_context, CudaIntDevice, Context, cublas::{create_handle, CublasHandle, cublasSetStream_v2, cublasDestroy_v2}, cuInit, cufree, cumalloc, cu_write, cu_read, cuCtxDestroy, Stream, create_stream, cuStreamDestroy, Module, cuModuleUnload}, CudaCache, cu_clear};

/// Used to perform calculations with a CUDA capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CudaDevice].
#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub inner: Rc<RefCell<InternCudaDevice>>
}

impl CudaDevice {
    pub fn new(idx: usize) -> crate::Result<CudaDevice> {
        let inner = Rc::new(RefCell::new(InternCudaDevice::new(idx)?));
        Ok(
            CudaDevice { inner }
        )
    }

    pub fn handle(&self) -> Ref<CublasHandle> {
        let borrow = self.inner.borrow();
        Ref::map(borrow, |x| &x.handle)    
    }

    pub fn stream(&self) -> RefMut<Stream> {
        let borrow = self.inner.borrow_mut();
        RefMut::map(borrow, |x| &mut x.stream)    
    }
}

#[cfg(not(feature="safe"))]
impl<T> Device<T> for CudaDevice {
    fn alloc(&self, len: usize) -> (*mut T, *mut std::ffi::c_void, u64) {
        let ptr = cumalloc::<T>(len).unwrap();
        self.inner.borrow_mut().ptrs.push(ptr);
        // TODO: use unified mem if available -> i can't test this
        (null_mut(), null_mut(), ptr)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut std::ffi::c_void, u64) {
        let ptr = cumalloc::<T>(data.len()).unwrap();
        self.inner.borrow_mut().ptrs.push(ptr);
        cu_write(ptr, data).unwrap();
        (null_mut(), null_mut(), ptr)
    }

    fn drop(&mut self, buf: crate::Buffer<T>) {
        let ptrs = &mut self.inner.borrow_mut().ptrs;
        crate::remove_value(ptrs, &buf.ptr.2).unwrap();
        unsafe {
            cufree(buf.ptr.2).unwrap();
        }
    }
}

#[cfg(feature="safe")]
impl<T> Device<T> for CudaDevice {
    fn alloc(&self, len: usize) -> (*mut T, *mut std::ffi::c_void, u64) {
        (null_mut(), null_mut(), cumalloc::<T>(len).unwrap())
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut std::ffi::c_void, u64) {
        let ptr = cumalloc::<T>(data.len()).unwrap();
        cu_write(ptr, data).unwrap();
        (null_mut(), null_mut(), ptr)
    }
}

impl<T: Default + Copy> VecRead<T> for CudaDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        assert!(buf.ptr.2 != 0, "called VecRead::read(..) on a non CUDA buffer");
        let mut read = vec![T::default(); buf.len];
        cu_read(&mut read, buf.ptr.2).unwrap();
        read
    }
}

impl<T: CDatatype> ClearBuf<T> for CudaDevice {
    fn clear(&self, buf: &mut crate::Buffer<T>) {
        cu_clear(self, buf).unwrap()
    }
}

impl<T> CacheBuf<T> for CudaDevice {
    fn cached_buf(&self, len: usize) -> crate::Buffer<T> {
        CudaCache::get::<T>(self, len)
    }
}

impl AsDev for CudaDevice {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(None, None, Some(Rc::downgrade(&self.inner)))
    }
}

impl<T: CDatatype + GenericBlas> BaseDevice<T> for CudaDevice {}

#[derive(Debug)]
pub struct InternCudaDevice {
    pub ptrs: Vec<CUdeviceptr>,
    pub modules: Vec<Module>,
    device: CudaIntDevice,
    ctx: Context,
    stream: Stream,
    handle: CublasHandle,
}

impl From<Rc<RefCell<InternCudaDevice>>> for CudaDevice {
    fn from(inner: Rc<RefCell<InternCudaDevice>>) -> Self {
        CudaDevice { inner }
    }
}

impl InternCudaDevice {
    #[must_use]
    pub fn new(idx: usize) -> crate::Result<InternCudaDevice> {
        unsafe {cuInit(0) }.to_result()?;
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let handle = create_handle()?;
        let stream = create_stream()?;
        unsafe {cublasSetStream_v2(handle.0, stream.0)}.to_result()?;
        
        Ok(InternCudaDevice {
            ptrs: vec![],
            modules: vec![],
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


impl Drop for InternCudaDevice {
    fn drop(&mut self) {
        unsafe {
            for ptr in &mut self.ptrs {
                cufree(*ptr).unwrap();
            }

            cublasDestroy_v2(self.handle.0);

            for module in &self.modules {
                cuModuleUnload(module.0);
            }

            cuStreamDestroy(self.stream.0);
            cuCtxDestroy(self.ctx.0);        
        }    
    }
}
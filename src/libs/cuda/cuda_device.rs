use std::{cell::{RefCell, Ref, RefMut}, rc::Rc, ptr::null_mut};
use crate::{Device, VecRead, CacheBuf, Gemm, BaseOps, AssignOps, BaseDevice, GenericBlas, CDatatype, Buffer, Matrix, CUdeviceptr, AsDev};
use super::{api::{device, create_context, CudaIntDevice, Context, cublas::{create_handle, CublasHandle, cublasSetStream_v2, cublasDestroy_v2}, cuInit, cufree, cumalloc, cuwrite, curead, cuCtxDestroy, Stream, create_stream, cuStreamDestroy, Module, cuModuleUnload}, CudaCache, cu_clear, cu_ew, cu_ew_self};

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
        cuwrite(ptr, data).unwrap();
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
        cuwrite(ptr, data).unwrap();
        (null_mut(), null_mut(), ptr)
    }

    fn drop(&mut self, buf: crate::Buffer<T>) {
        unsafe {
            cufree(buf.ptr.2).unwrap();
        }
    }
}

impl<T: Default + Copy> VecRead<T> for CudaDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        let mut read = vec![T::default(); buf.len];
        curead(&mut read, buf.ptr.2).unwrap();
        read
    }
}

impl<T> CacheBuf<T> for CudaDevice {
    fn cached_buf(&self, len: usize) -> crate::Buffer<T> {
        CudaCache::get::<T>(self, len)
    }
}

impl<T: GenericBlas> Gemm<T> for CudaDevice {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(lhs.cols() == rhs.rows(), "wrong dims for matrix multiplication");
        let out: Buffer<T> = self.cached_buf(lhs.rows() * rhs.cols());
        T::cugemm(
            &self.inner.borrow().handle, 
            lhs.rows(), 
            rhs.cols(), 
            lhs.cols(), 
            lhs.as_buf().ptr.2, 
            rhs.as_buf().ptr.2, 
            out.ptr.2
        ).unwrap();
        (out, lhs.rows(), rhs.cols()).into()
    }
}

impl<T: CDatatype> AssignOps<T> for CudaDevice {
    fn add_assign(&self, lhs: &mut crate::Buffer<T>, rhs: &crate::Buffer<T>) {
        cu_ew_self(self, lhs, rhs, "+").unwrap();
    }

    fn sub_assign(&self, lhs: &mut crate::Buffer<T>, rhs: &crate::Buffer<T>) {
        cu_ew_self(self, lhs, rhs, "-").unwrap();
    }
}

impl<T: CDatatype> BaseOps<T> for CudaDevice {
    fn add(&self, lhs: &crate::Matrix<T>, rhs: &crate::Matrix<T>) -> crate::Matrix<T> {
        let buf = cu_ew(self, lhs, rhs, "+").unwrap();
        (buf, lhs.dims()).into()
    }

    fn sub(&self, lhs: &crate::Matrix<T>, rhs: &crate::Matrix<T>) -> crate::Matrix<T> {
        let buf = cu_ew(self, lhs, rhs, "-").unwrap();
        (buf, lhs.dims()).into()
    }

    fn mul(&self, lhs: &crate::Matrix<T>, rhs: &crate::Matrix<T>) -> crate::Matrix<T> {
        let buf = cu_ew(self, lhs, rhs, "*").unwrap();
        (buf, lhs.dims()).into()
    }

    fn div(&self, lhs: &crate::Matrix<T>, rhs: &crate::Matrix<T>) -> crate::Matrix<T> {
        let buf = cu_ew(self, lhs, rhs, "/").unwrap();
        (buf, lhs.dims()).into()
    }

    fn clear(&self, buf: &mut crate::Buffer<T>) {
        cu_clear(self, buf).unwrap();
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
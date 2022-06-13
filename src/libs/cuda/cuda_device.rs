use std::{cell::RefCell, rc::Rc, ptr::null_mut};
use crate::{Device, remove_value, VecRead, CacheBuf, Gemm, BaseOps, AssignOps, BaseDevice, GenericBlas, GenericOCL, Buffer, Matrix, CUdeviceptr};
use super::{api::{device, create_context, CudaIntDevice, Context, cublas::{create_handle, CublasHandle}, cuInit, cufree, cumalloc, cuwrite, curead}, CudaCache};

#[derive(Debug, Clone)]
pub struct InternCudaDevice {
    pub cuda: Rc<RefCell<CudaDevice>>
}

impl From<Rc<RefCell<CudaDevice>>> for InternCudaDevice {
    fn from(cuda: Rc<RefCell<CudaDevice>>) -> Self {
        InternCudaDevice { cuda }
    }
}

impl InternCudaDevice {
    #[must_use]
    pub fn new(cuda: CudaDevice) -> InternCudaDevice {
        let cuda = Rc::new(RefCell::new(cuda));
        InternCudaDevice { cuda }
    }
}

impl<T> Device<T> for InternCudaDevice {
    fn alloc(&self, len: usize) -> (*mut T, *mut std::ffi::c_void, u64) {
        let ptr = cumalloc::<T>(len).unwrap();
        self.cuda.borrow_mut().ptrs.push(ptr);
        // TODO: use unified mem if available -> i can't test this
        (null_mut(), null_mut(), ptr)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut std::ffi::c_void, u64) {
        let ptr = cumalloc::<T>(data.len()).unwrap();
        self.cuda.borrow_mut().ptrs.push(ptr);
        cuwrite(ptr, data).unwrap();
        (null_mut(), null_mut(), ptr)
    }

    fn drop(&mut self, buf: crate::Buffer<T>) {
        let ptrs = &mut self.cuda.borrow_mut().ptrs;
        remove_value(ptrs, &buf.ptr.2).unwrap();
        unsafe {
            cufree(buf.ptr.2).unwrap();
        }
    }
}

impl<T: Default + Copy> VecRead<T> for InternCudaDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        let mut read = vec![T::default(); buf.len];
        curead(&mut read, buf.ptr.2).unwrap();
        read
    }
}

impl<T> CacheBuf<T> for InternCudaDevice {
    fn cached_buf(&self, len: usize) -> crate::Buffer<T> {
        CudaCache::get::<T>(self.clone(), len)
    }
}

impl<T: GenericBlas> Gemm<T> for InternCudaDevice {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(lhs.cols() == rhs.rows(), "wrong dims for matrix multiplication");
        let out: Buffer<T> = self.cached_buf(lhs.rows() * rhs.cols());
        T::cugemm(
            &self.cuda.borrow().handle, 
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

impl<T> AssignOps<T> for InternCudaDevice {
    fn add_assign(&self, lhs: &mut crate::Matrix<T>, rhs: &crate::Matrix<T>) {
        todo!()
    }

    fn sub_assign(&self, lhs: &mut crate::Matrix<T>, rhs: &crate::Matrix<T>) {
        todo!()
    }
}

impl<T> BaseOps<T> for InternCudaDevice {
    fn add(&self, lhs: &crate::Matrix<T>, rhs: &crate::Matrix<T>) -> crate::Matrix<T> {
        todo!()
    }

    fn sub(&self, lhs: &crate::Matrix<T>, rhs: &crate::Matrix<T>) -> crate::Matrix<T> {
        todo!()
    }

    fn mul(&self, lhs: &crate::Matrix<T>, rhs: &crate::Matrix<T>) -> crate::Matrix<T> {
        todo!()
    }

    fn div(&self, lhs: &crate::Matrix<T>, rhs: &crate::Matrix<T>) -> crate::Matrix<T> {
        todo!()
    }

    fn clear(&self, matrix: &mut crate::Matrix<T>) {
        todo!()
    }
}

impl<T: GenericOCL + GenericBlas> BaseDevice<T> for InternCudaDevice {}

#[derive(Debug)]
pub struct CudaDevice {
    pub ptrs: Vec<CUdeviceptr>,
    device: CudaIntDevice,
    ctx: Context,
    handle: CublasHandle

}

impl CudaDevice {
    pub fn new(idx: usize) -> crate::Result<InternCudaDevice> {
        unsafe {cuInit(0) };
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let handle = create_handle().unwrap();

        let device = CudaDevice {
            ptrs: vec![],
            device,
            ctx,
            handle,
        };
        Ok(InternCudaDevice::new(device))

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
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        for ptr in &mut self.ptrs {
            unsafe {
                cufree(*ptr).unwrap();
            }
        }
    }
}
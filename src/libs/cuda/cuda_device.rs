use std::{cell::RefCell, rc::Rc, ptr::null_mut};

use crate::{Device, remove_value};

use super::api::{device, create_context, CudaIntDevice, Context, cublas::{create_handle, CublasHandle}, cuInit, CUdeviceptr, cufree, cumalloc, cuwrite};

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
use std::{cell::RefCell, rc::Rc};

use super::api::{device, create_context, CudaIntDevice, Context, cublas::{create_handle, CublasHandle}, cuInit, CUdeviceptr};

pub struct InternCudaDevice {
    pub cuda: Rc<RefCell<CudaDevice>>
}

pub struct CudaDevice {
    pub ptrs: Vec<CUdeviceptr>,
    device: CudaIntDevice,
    ctx: Context,
    handle: CublasHandle

}

impl CudaDevice {
    pub fn new(idx: usize) -> crate::Result<CudaDevice> {
        unsafe {cuInit(0) };
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let handle = create_handle().unwrap();

        Ok(CudaDevice {
            ptrs: vec![],
            device,
            ctx,
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
}
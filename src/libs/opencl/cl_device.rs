use std::ffi::c_void;

use crate::{buffer::Device, libs::opencl::api::{MemFlags, create_buffer}, VecRead, BaseDevice, AsDev, matrix::Matrix, BaseOps, Gemm};

use super::{api::{Context, CommandQueue, OCLError, create_context, create_command_queue, CLIntDevice, wait_for_event, enqueue_read_buffer}, CL_DEVICES, tew, GenericOCL, ocl_gemm};


#[derive(Debug, Clone, Copy)]
pub struct CLDevice {
    pub device: CLIntDevice,
    pub ctx: Context,
    pub queue: CommandQueue,
}

impl CLDevice {
    pub fn new(device: CLIntDevice) -> Result<CLDevice, OCLError> {
        let ctx = create_context(&[device])?;
        let queue = create_command_queue(&ctx, device)?;

        Ok(CLDevice {
                device,
                ctx,
                queue
        })
        
    }

    pub fn get(device_idx: usize) -> Result<CLDevice, OCLError>{
        unsafe {CL_DEVICES.get_current(device_idx)}
    }

    pub fn get_ctx(&self) -> &Context {
        &self.ctx
    }
    pub fn get_queue(&self) -> CommandQueue {
        self.queue
    }
    pub fn get_global_mem_size_in_gb(&self) -> Result<f64, OCLError> {
        Ok(self.device.get_global_mem()? as f64 * 10f64.powi(-9))
    }
    pub fn get_max_mem_alloc_in_gb(&self) -> Result<f64, OCLError> {
        Ok(self.device.get_max_mem_alloc()? as f64 * 10f64.powi(-9))
    }
    pub fn get_name(&self) -> Result<String, OCLError> {
        Ok(self.device.get_name()?)
    }
}

impl <T: GenericOCL>Gemm<T> for CLDevice {
    fn gemm(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ocl_gemm(*self, rhs, lhs).unwrap()
    }
}

impl <T: GenericOCL>BaseDevice<T> for CLDevice {}

impl <T: GenericOCL>BaseOps<T> for CLDevice {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        tew(*self, lhs, rhs, "+").unwrap()
    }
}


impl <T>Device<T> for CLDevice {
    fn alloc(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn from_data(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap() as *mut T
    }
}



impl AsDev for CLDevice {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(Some(self.clone()))
    }
}

/* 

impl Device for CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn from_data<T>(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap() as *mut T
    }
}

impl Device for &mut CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn from_data<T>(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap() as *mut T
    }
}

impl Device for &CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn from_data<T>(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap() as *mut T
    }
}

*/

impl <T: Default+Copy>VecRead<T> for CLDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        let mut read = vec![T::default(); buf.len];
        let event = enqueue_read_buffer(&self.get_queue(), buf.ptr as *mut c_void, &mut read, true).unwrap();
        wait_for_event(event).unwrap();
        read
    }
}

/*

impl <T: Default+Copy>VecRead<T> for &mut CLDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        let mut read = vec![T::default(); buf.len];
        let event = enqueue_read_buffer(&self.get_queue(), buf.ptr as *mut c_void, &mut read, true).unwrap();
        wait_for_event(event).unwrap();
        read
    }
}
*/

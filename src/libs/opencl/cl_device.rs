use std::ffi::c_void;

use crate::{buffer::Device, libs::opencl::api::{MemFlags, create_buffer}, VecRead};

use super::{api::{Context, CommandQueue, OCLError, create_context, create_command_queue, CLIntDevice, wait_for_event, enqueue_read_buffer}, CL_DEVICES};


#[derive(Debug,)]
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

    pub fn get<'a>(device_idx: usize) -> Result<&'a mut CLDevice, OCLError>{
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

impl Device for CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn from_data<T>(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, data.len(), Some(data)).unwrap() as *mut T
    }
}

impl Device for &mut CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn from_data<T>(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, data.len(), Some(data)).unwrap() as *mut T
    }
}

impl Device for &CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn from_data<T>(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, data.len(), Some(data)).unwrap() as *mut T
    }
}

impl <T: Default+Copy>VecRead<T> for &CLDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        let mut read = vec![T::default(); buf.len];
        let event = enqueue_read_buffer(&self.get_queue(), buf.ptr as *mut c_void, &mut read, true).unwrap();
        wait_for_event(event).unwrap();
        read
    }
}


use crate::{buffer::Alloc, libs::opencl::api::{MemFlags, create_buffer}};

use super::{api::{Device, Context, CommandQueue, OCLError, create_context, create_command_queue}, CL_DEVICES};


#[derive(Debug,)]
pub struct CLDevice {
    pub device: Device,
    pub ctx: Context,
    pub queue: CommandQueue,
}

impl CLDevice {
    pub fn new(device: Device) -> Result<CLDevice, OCLError> {
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

impl Alloc for CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }
}

impl Alloc for &mut CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }
}

impl Alloc for &CLDevice {
    fn alloc<T>(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }
}
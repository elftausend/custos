use std::{ffi::c_void, rc::Rc, cell::RefCell};

use crate::{buffer::Device, libs::opencl::api::{create_buffer, MemFlags}, BaseOps, Matrix, AsDev, Gemm, VecRead, BaseDevice};

use super::{api::{CLIntDevice, CommandQueue, Context, create_command_queue, create_context, OCLError, enqueue_read_buffer, wait_for_event}, CL_DEVICES2, GenericOCL, tew, ocl_gemm};

#[derive(Debug, Clone)]
pub struct InternCLDevice {
    pub cl: Rc<RefCell<CLDevice>>
}

impl InternCLDevice {
    pub fn new(cl: CLDevice) -> InternCLDevice {
        let cl = Rc::new(RefCell::new(cl));
         InternCLDevice { cl }
    }

    pub fn get_ctx(&self) -> Context {
        self.cl.borrow().ctx
    }

    pub fn get_queue(&self) -> CommandQueue {
        self.cl.borrow().queue
    }

    pub fn device(&self) -> CLIntDevice {
        self.cl.borrow().device
    }

    pub fn get_global_mem_size_in_gb(&self) -> Result<f64, OCLError> {
        Ok(self.device().get_global_mem()? as f64 * 10f64.powi(-9))
    }
    pub fn get_max_mem_alloc_in_gb(&self) -> Result<f64, OCLError> {
        Ok(self.device().get_max_mem_alloc()? as f64 * 10f64.powi(-9))
    }
    pub fn get_name(&self) -> Result<String, OCLError> {
        self.device().get_name()
    }
    pub fn get_version(&self) -> Result<String, OCLError> {
        self.device().get_version()
    }
}

impl <T>Device<T> for InternCLDevice {
    fn alloc(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn with_data(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.get_ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap() as *mut T
    }
}


impl <T: GenericOCL>BaseOps<T> for InternCLDevice {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        tew(self.clone(), lhs, rhs, "+").unwrap()
    }

    fn sub(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        tew(self.clone(), lhs, rhs, "-").unwrap()
    }

    fn mul(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        tew(self.clone(), lhs, rhs, "*").unwrap()
    }
}

impl <T: GenericOCL>Gemm<T> for InternCLDevice {
    fn gemm(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ocl_gemm(self.clone(), rhs, lhs).unwrap()   
    }
}

impl <T: Default+Copy>VecRead<T> for InternCLDevice {
    fn read(&self, buf: crate::Buffer<T>) -> Vec<T> {
        let mut read = vec![T::default(); buf.len];
        let event = enqueue_read_buffer(&self.get_queue(), buf.ptr as *mut c_void, &mut read, true).unwrap();
        wait_for_event(event).unwrap();
        read
    }
}

impl AsDev for InternCLDevice {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(Some(self.clone()), None)
    }
}

impl <T: GenericOCL>BaseDevice<T> for InternCLDevice {}


#[derive(Debug, Clone)]
pub struct CLDevice {
    pub ptrs: Vec<*mut c_void>,
    pub device: CLIntDevice,
    pub ctx: Context,
    pub queue: CommandQueue,
}

unsafe impl Sync for CLDevice {}
unsafe impl Send for CLDevice {}

impl CLDevice {
    pub fn new(device: CLIntDevice) -> Result<CLDevice, OCLError> {
        let ctx = create_context(&[device])?;
        let queue = create_command_queue(&ctx, device)?;

        Ok(CLDevice { ptrs: Vec::new(), device, ctx, queue }) 
    }

    pub fn get(device_idx: usize) -> Result<InternCLDevice, OCLError> {
        Ok(InternCLDevice::new(CL_DEVICES2.get_current(device_idx)?))
    }
}

/* 

#[derive(Debug, Clone, Copy)]
pub struct CLDevice {
    pub device: CLIntDevice,
    pub ctx: Context,
    pub queue: CommandQueue,
}

unsafe impl Sync for CLDevice {}
unsafe impl Send for CLDevice {}

impl CLDevice {
    pub fn new(device: CLIntDevice) -> Result<CLDevice, OCLError> {
        let ctx = create_context(&[device])?;
        let queue = create_command_queue(&ctx, device)?;

        Ok(CLDevice { device, ctx, queue }) 
    }

    pub fn get(device_idx: usize) -> Result<CLDevice, OCLError> {
        CL_DEVICES.get_current(device_idx)
    }

    pub fn mt<T: Default+Copy>(device_idx: usize) -> Result<(Self, Threaded<CLDevice>), OCLError> {
        let device = CLDevice::get(device_idx)?;
        Ok((device, Threaded::new(device)))
    }
    pub fn drop<T>(buffer: crate::Buffer<T>) {
        release_mem_object(buffer.ptr as *mut c_void).unwrap();
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
        self.device.get_name()
    }
    pub fn get_version(&self) -> Result<String, OCLError> {
        self.device.get_version()
    }
    
}

impl Dealloc for CLDevice {
    fn dealloc_cache() {
        CL_CACHE.with(|cache| {
            let contents = cache.borrow().output_nodes.clone();
            contents.into_iter()
                .for_each(|entry| {
                    let ptr = (entry.1).0;
                    release_mem_object(ptr.0).unwrap();
                    cache.borrow_mut().output_nodes.remove(&entry.0);
                });
        });
        /*
        let mut cache = CL_CACHE.lock().unwrap();
        
        let contents = cache.output_nodes.clone();
        for entry in contents.into_iter() {
            if entry.0.thread_id == std::thread::current().id() {
                let ptr = (entry.1).0;
                release_mem_object(ptr.0).unwrap();
                cache.output_nodes.remove(&entry.0);
           }    
        };
        */
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

    fn sub(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        tew(*self, lhs, rhs, "-").unwrap()
    }

    fn mul(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        tew(*self, lhs, rhs, "*").unwrap()
    }
}


impl <T>Device<T> for CLDevice {
    fn alloc(&self, len: usize) -> *mut T {
        create_buffer::<T>(self.get_ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn with_data(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(self.get_ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap() as *mut T
    }
}



impl AsDev for CLDevice {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(Some(*self))
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
    fn read(&self, buf: crate::Buffer<T>) -> Vec<T> {
        let mut read = vec![T::default(); buf.len];
        let event = enqueue_read_buffer(&self.get_queue(), buf.ptr as *mut c_void, &mut read, true).unwrap();
        wait_for_event(event).unwrap();
        read
    }
}
*/
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

use std::{ffi::c_void, rc::Rc, cell::RefCell};

use crate::{libs::opencl::api::{create_buffer, MemFlags}, BaseOps, Matrix, AsDev, Gemm, VecRead, BaseDevice, Error, Device};

use super::{api::{CLIntDevice, CommandQueue, Context, create_command_queue, create_context, enqueue_read_buffer, wait_for_event, release_mem_object}, CL_DEVICES, GenericOCL, tew, ocl_gemm, CL_CACHE};

#[derive(Debug, Clone)]
pub struct InternCLDevice {
    pub cl: Rc<RefCell<CLDevice>>
}

impl From<Rc<RefCell<CLDevice>>> for InternCLDevice {
    fn from(cl: Rc<RefCell<CLDevice>>) -> Self {
        InternCLDevice { cl }
    }
}

impl InternCLDevice {
    #[must_use]
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

    pub fn get_global_mem_size_in_gb(&self) -> Result<f64, Error> {
        Ok(self.device().get_global_mem()? as f64 * 10f64.powi(-9))
    }
    pub fn get_max_mem_alloc_in_gb(&self) -> Result<f64, Error> {
        Ok(self.device().get_max_mem_alloc()? as f64 * 10f64.powi(-9))
    }
    pub fn get_name(&self) -> Result<String, Error> {
        self.device().get_name()
    }
    pub fn get_version(&self) -> Result<String, Error> {
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
        crate::Dev::new(Some(Rc::downgrade(&self.cl)), None)
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

impl CLDevice {
    pub fn new(device: CLIntDevice) -> Result<CLDevice, Error> {
        let ctx = create_context(&[device])?;
        let queue = create_command_queue(&ctx, device)?;

        Ok(CLDevice { ptrs: Vec::new(), device, ctx, queue }) 
    }

    pub fn get(device_idx: usize) -> Result<InternCLDevice, Error> {
        Ok(InternCLDevice::new(CL_DEVICES.get_current(device_idx)?))
    }
}

impl Drop for CLDevice {
    fn drop(&mut self) {
        let contents = CL_CACHE.with(|cache| {
           cache.borrow().output_nodes.clone()         
        });

        for ptr in self.ptrs.iter() {

            release_mem_object(*ptr).unwrap();

            contents.iter()
                .for_each(|entry| {
                    let hm_ptr = ((entry.1).0).0;

                    if &hm_ptr == ptr {
                        CL_CACHE.with(|cache| {
                            cache.borrow_mut().output_nodes.remove(entry.0);
                        });                        
                    }
                });
        }

        self.ptrs.clear();
    }
}

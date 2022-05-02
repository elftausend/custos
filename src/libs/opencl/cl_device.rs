use std::{ffi::c_void, rc::Rc, cell::RefCell};

use crate::{libs::opencl::api::{create_buffer, MemFlags}, BaseOps, Matrix, AsDev, Gemm, VecRead, BaseDevice, Error, Device, AssignOps, GenericOCL, DropBuf};

use super::{api::{CLIntDevice, CommandQueue, Context, create_command_queue, create_context, enqueue_read_buffer, wait_for_event, release_mem_object, enqueue_write_buffer}, CL_DEVICES, tew, ocl_gemm, CL_CACHE, tew_self};

#[derive(Debug, Clone)]
/// All traits related to mathematical operations need to be implemented for this struct in order to use them.
/// This struct is should be only created via the [CLDevice] struct.
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

    pub fn ctx(&self) -> Context {
        self.cl.borrow().ctx
    }

    pub fn queue(&self) -> CommandQueue {
        self.cl.borrow().queue
    }

    pub fn device(&self) -> CLIntDevice {
        self.cl.borrow().device
    }

    pub fn global_mem_size_in_gb(&self) -> Result<f64, Error> {
        Ok(self.device().get_global_mem()? as f64 * 10f64.powi(-9))
    }

    pub fn max_mem_alloc_in_gb(&self) -> Result<f64, Error> {
        Ok(self.device().get_max_mem_alloc()? as f64 * 10f64.powi(-9))
    }

    pub fn name(&self) -> Result<String, Error> {
        self.device().get_name()
    }

    pub fn version(&self) -> Result<String, Error> {
        self.device().get_version()
    }

    pub fn unified_mem(&self) -> Result<bool, Error> {
        self.device().unified_mem()
    }
}

#[cfg(not(feature="safe"))]
impl<T> Device<T> for InternCLDevice {
    fn alloc(&self, len: usize) -> *mut T {
        let ptr = create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T;
        self.cl.borrow_mut().ptrs.push(ptr as *mut c_void);
        ptr
    }

    fn with_data(&self, data: &[T]) -> *mut T {
        let ptr = create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap() as *mut T;
        self.cl.borrow_mut().ptrs.push(ptr as *mut c_void);
        ptr
    }
}

#[cfg(feature="safe")]
impl<T> Device<T> for InternCLDevice {
    fn alloc(&self, len: usize) -> *mut T {
        create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap() as *mut T
    }

    fn with_data(&self, data: &[T]) -> *mut T {
        create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap() as *mut T
    }

    fn dealloc_type(&self) -> crate::DeallocType {
        crate::DeallocType::CL
    }
}

impl<T> DropBuf<T> for InternCLDevice {
    fn drop_buf(&self, buf: &mut crate::Buffer<T>) {
        release_mem_object(buf.ptr as *mut c_void).unwrap();
    }
}

impl<T: GenericOCL> BaseOps<T> for InternCLDevice {
    fn add(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        tew(self.clone(), lhs, rhs, "+").unwrap()
    }

    fn sub(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        tew(self.clone(), lhs, rhs, "-").unwrap()
    }

    fn mul(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        tew(self.clone(), lhs, rhs, "*").unwrap()
    }

    fn div(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        tew(self.clone(), lhs, rhs, "/").unwrap()
    }
}

impl<T: GenericOCL> AssignOps<T> for InternCLDevice {
    fn sub_assign(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>) {
        tew_self(self.clone(), lhs, rhs, "-").unwrap()
    }
}

impl<T: GenericOCL> Gemm<T> for InternCLDevice {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(lhs.dims().1 == rhs.dims().0);
        ocl_gemm(self.clone(), rhs, lhs).unwrap()   
    }
}

pub fn cl_write<T>(device: &InternCLDevice, x: &mut Matrix<T>, data: &[T]) {
    let event = enqueue_write_buffer(&device.queue(), x.ptr() as *mut c_void, data, true).unwrap();
    wait_for_event(event).unwrap();
} 

impl<T: Default+Copy> VecRead<T> for InternCLDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        let mut read = vec![T::default(); buf.len];
        let event = unsafe {enqueue_read_buffer(&self.queue(), buf.ptr as *mut c_void, &mut read, true).unwrap()};
        wait_for_event(event).unwrap();
        read
    }
}

impl AsDev for InternCLDevice {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(Some(Rc::downgrade(&self.cl)), None)
    }
}

impl<T: GenericOCL> BaseDevice<T> for InternCLDevice {}

#[derive(Debug, Clone)]
/// If the 'safe' feature isn't used, pointers are stored in the 'ptrs' field.
/// It is used to get an [InternCLDevice], which gives you access to all functions that were implemented for the InternCLDevice struct.
/// 
/// # Note / Safety
/// 
/// If the 'safe' feature isn't used, all pointers will get invalid when the drop code for a CLDevice object is run as that deallocates the memory previously pointed at by the pointers stored in 'ptrs'.
/// 
/// # Example
/// ```
/// use custos::{CLDevice, BaseOps, VecRead, Matrix, Error};
/// 
/// fn main() -> Result<(), Error> {
///     let device = CLDevice::get(0)?;
///     
///     let a = Matrix::<f32>::new(&device, (5, 5));
///     let b = Matrix::from((&device, (5, 5), vec![1.3; 5*5]));
///     
///     let out = device.add(&a, &b);
///     
///     assert_eq!(device.read(out.data()), vec![1.3; 5*5]);
///     Ok(())
/// }
/// ```
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
    /// Returns an [InternCLDevice] at the specified device index.
    /// # Errors
    /// - No device is found at the given device index
    /// - some other OpenCL related errors
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

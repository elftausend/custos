use std::{ffi::c_void, rc::Rc, cell::RefCell};
use crate::{libs::opencl::api::{create_buffer, MemFlags}, BaseOps, Matrix, AsDev, Gemm, VecRead, BaseDevice, Error, Device, AssignOps, CDatatype, ManualMem, Buffer, CacheBuf};
use super::{api::{CLIntDevice, CommandQueue, Context, create_command_queue, create_context, enqueue_read_buffer, wait_for_event, release_mem_object, enqueue_write_buffer, unified_ptr}, CL_DEVICES, cl_tew, cl_gemm, CL_CACHE, cl_tew_self, CLCache, cl_clear};

#[derive(Debug, Clone)]
/// Used to perform calculations with an OpenCL capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CLDevice].
/// # Example
/// ```
/// use custos::{CLDevice, BaseOps, VecRead, Matrix, Error};
/// 
/// fn main() -> Result<(), Error> {
///     let device = CLDevice::new(0)?;
///     
///     let a = Matrix::<f32>::new(&device, (5, 5));
///     let b = Matrix::from((&device, (5, 5), vec![1.3; 5*5]));
///     
///     let out = device.add(&a, &b);
///     
///     assert_eq!(device.read(&out), vec![1.3; 5*5]);
///     Ok(())
/// }
/// ```
pub struct CLDevice {
    pub inner: Rc<RefCell<InternCLDevice>>
}

unsafe impl Sync for InternCLDevice {}

impl CLDevice {
    /// Returns an [CLDevice] at the specified device index.
    /// # Errors
    /// - No device is found at the given device index
    /// - some other OpenCL related errors
    pub fn new(device_idx: usize) -> Result<CLDevice, Error> {
        let inner = Rc::new(RefCell::new(CL_DEVICES.current(device_idx)?));
        Ok(
            CLDevice { inner }
        )
    }

    pub fn ctx(&self) -> Context {
        self.inner.borrow().ctx
    }

    pub fn queue(&self) -> CommandQueue {
        self.inner.borrow().queue
    }

    pub fn device(&self) -> CLIntDevice {
        self.inner.borrow().device
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

    pub fn unified_mem(&self) -> bool {
        // TODO: "true" for every device?
        self.inner.borrow().unified_mem
    }
}


#[cfg(not(feature="safe"))]
impl<T> Device<T> for CLDevice {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        let ptr = create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap();
        self.inner.borrow_mut().ptrs.push(ptr);

        let cpu_ptr = if self.unified_mem() {
            unified_ptr::<T>(self.queue(), ptr, len).unwrap()
        } else {
            std::ptr::null_mut()
        };

        (cpu_ptr, ptr, 0)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64) {
        let ptr = create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap();
        self.inner.borrow_mut().ptrs.push(ptr as *mut c_void);

        let cpu_ptr = if self.unified_mem() {
            unified_ptr::<T>(self.queue(), ptr, data.len()).unwrap()
        } else {
            std::ptr::null_mut()
        };

        (cpu_ptr, ptr, 0)
    }

    fn drop(&mut self, buf: Buffer<T>) {
        let ptrs = &mut self.inner.borrow_mut().ptrs;
        crate::remove_value(ptrs, &buf.ptr.1).unwrap();
        self.drop_buf(buf)
    }
}

#[cfg(feature="safe")]
impl<T> Device<T> for CLDevice {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        let ptr = create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap();
        let cpu_ptr = if self.unified_mem() {
            unified_ptr::<T>(self.queue(), ptr, len).unwrap()
        } else {
            std::ptr::null_mut()
        };
        (cpu_ptr, ptr, 0)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64) {
        let ptr = create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr, data.len(), Some(data)).unwrap();
        let cpu_ptr = if self.unified_mem() {
            unified_ptr::<T>(self.queue(), ptr, data.len()).unwrap()
        } else {
            std::ptr::null_mut()
        };
        (cpu_ptr, ptr, 0)
    }

    fn drop(&mut self, buf: Buffer<T>) {
        self.drop_buf(buf)
    }
}

impl<T> ManualMem<T> for CLDevice {
    fn drop_buf(&self, buf: crate::Buffer<T>) {
        unsafe {
            release_mem_object(buf.ptr.1).unwrap();
        }
    }
}

impl<T: CDatatype> CacheBuf<T> for CLDevice {
    fn cached_buf(&self, len: usize) -> Buffer<T> {
        CLCache::get::<T>(self, len)
    }
}

impl<T: CDatatype> BaseOps<T> for CLDevice {
    fn add(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        let buf = cl_tew(self, lhs, rhs, "+").unwrap();
        (buf, lhs.dims()).into()

    }

    fn sub(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        let buf = cl_tew(self, lhs, rhs, "-").unwrap();
        (buf, lhs.dims()).into()
    }

    fn mul(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        let buf = cl_tew(self, lhs, rhs, "*").unwrap();
        (buf, lhs.dims()).into()
    }

    fn div(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        let buf = cl_tew(self, lhs, rhs, "/").unwrap();
        (buf, lhs.dims()).into()
    }

    fn clear(&self, buf: &mut Buffer<T>) {
        cl_clear(self, buf).unwrap();
    }
}

impl<T: CDatatype> AssignOps<T> for CLDevice {
    fn add_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        cl_tew_self(self, lhs, rhs, "+").unwrap()
    }

    fn sub_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        cl_tew_self(self, lhs, rhs, "-").unwrap()
    }
}

impl<T: CDatatype> Gemm<T> for CLDevice {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(lhs.dims().1 == rhs.dims().0);
        //crate::opencl::ops::ocl_gemm1(self.clone(), rhs, lhs).unwrap()
        let buf = cl_gemm(self, rhs.cols(), rhs.rows(), lhs.rows(), rhs, lhs).unwrap();
        (buf, (lhs.rows(), rhs.cols())).into()
    }
}

pub fn cl_write<T>(device: &CLDevice, x: &mut Matrix<T>, data: &[T]) {
    let event = unsafe {enqueue_write_buffer(&device.queue(), x.ptr().1, data, true).unwrap()};
    wait_for_event(event).unwrap();
} 

impl<T: Default+Copy> VecRead<T> for CLDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        // TODO: check null?
        assert!(!buf.ptr.1.is_null(), "called VecRead::read(..) on a non OpenCL buffer (this would read out a null pointer)");
        let mut read = vec![T::default(); buf.len];
        let event = unsafe {enqueue_read_buffer(&self.queue(), buf.ptr.1, &mut read, false).unwrap()};
        wait_for_event(event).unwrap();
        read
    }
}

impl AsDev for CLDevice {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(Some(Rc::downgrade(&self.inner)), None, None)
    }
}

impl<T: CDatatype> BaseDevice<T> for CLDevice {}


#[derive(Debug, Clone)]
/// Internal representation of an OpenCL Device with the capability of storing pointers.
/// # Note / Safety
/// 
/// If the 'safe' feature isn't used, all pointers will get invalid when the drop code for a CLDevice object is run as that deallocates the memory previously pointed at by the pointers stored in 'ptrs'.
pub struct InternCLDevice {
    pub ptrs: Vec<*mut c_void>,
    device: CLIntDevice,
    ctx: Context,
    queue: CommandQueue,
    unified_mem: bool,
}

impl From<Rc<RefCell<InternCLDevice>>> for CLDevice {
    fn from(inner: Rc<RefCell<InternCLDevice>>) -> Self {
        CLDevice { inner }
    }
}

impl InternCLDevice {
    #[must_use]
    pub fn new(device: CLIntDevice) -> crate::Result<InternCLDevice> {
        let ctx = create_context(&[device])?;
        let queue = create_command_queue(&ctx, device)?;
        let unified_mem = device.unified_mem()?;

        Ok(InternCLDevice { ptrs: Vec::new(), device, ctx, queue, unified_mem })    
    }

}

impl Drop for InternCLDevice {
    fn drop(&mut self) {
        let contents = CL_CACHE.with(|cache| {
            /* 
            // FIXME: releases all kernels, even if it is used by another device?
            // TODO: better kernel cache release
            for kernel in &mut cache.borrow_mut().arg_kernel_cache.values_mut() {
                kernel.release()
            }
            */
            cache.borrow().nodes.clone()  
        });
        
        for ptr in self.ptrs.iter() {
            
            unsafe { release_mem_object(*ptr).unwrap() };

            for entry in &contents {
                let hm_ptr = ((entry.1).0).0;

                if &hm_ptr == ptr {
                    CL_CACHE.with(|cache| {
                        cache.borrow_mut().nodes.remove(entry.0);
                    });
                }
            }
        }
        self.ptrs.clear();
    }
}

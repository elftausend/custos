use super::{
    api::{
        create_command_queue, create_context, enqueue_read_buffer, enqueue_write_buffer,
        release_mem_object, unified_ptr, wait_for_event, CLIntDevice, CommandQueue, Context,
    },
    cl_clear, CLCache, CL_DEVICES,
};
use crate::{
    deallocate_cache, get_device_count,
    libs::opencl::api::{create_buffer, MemFlags},
    AsDev, BaseDevice, Buffer, CDatatype, CacheBuf, ClearBuf, Device, Error, ManualMem, VecRead,
    WriteBuf,
};
use std::{cell::RefCell, ffi::c_void, fmt::Debug, rc::Rc};

#[derive(Clone)]
/// Used to perform calculations with an OpenCL capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CLDevice].
/// # Example
/// ```
/// use custos::{CLDevice, VecRead, Buffer, Error};
///
/// fn main() -> Result<(), Error> {
///     let device = CLDevice::new(0)?;
///     
///     let a = Buffer::from((&device, [1.3; 25]));
///     let out = device.read(&a);
///     
///     assert_eq!(out, vec![1.3; 5*5]);
///     Ok(())
/// }
/// ```
pub struct CLDevice {
    pub inner: Rc<RefCell<InternCLDevice>>,
}

unsafe impl Sync for InternCLDevice {}

impl CLDevice {
    /// Returns an [CLDevice] at the specified device index.
    /// # Errors
    /// - No device is found at the given device index
    /// - some other OpenCL related errors
    pub fn new(device_idx: usize) -> Result<CLDevice, Error> {
        unsafe {
            *get_device_count() += 1;
        }
        let inner = Rc::new(RefCell::new(CL_DEVICES.current(device_idx)?));
        Ok(CLDevice { inner })
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
        self.inner.borrow().unified_mem
    }

    pub fn set_unified_mem(&self, unified_mem: bool) {
        self.inner.borrow_mut().unified_mem = unified_mem;
    }
}

impl Debug for CLDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CLDevice {{
            name: {name:?},
            version: {version:?},
            max_mem_alloc_in_gb: {max_mem:?},
            unified_mem: {unified_mem},
        }}",
            name = self.name(),
            version = self.version(),
            unified_mem = self.unified_mem(),
            max_mem = self.max_mem_alloc_in_gb()
        )
    }
}

impl<T> Device<T> for CLDevice {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        let ptr =
            create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap();

        #[cfg(not(feature = "safe"))]
        self.inner.borrow_mut().ptrs.push(ptr);

        let cpu_ptr = if self.unified_mem() {
            unified_ptr::<T>(self.queue(), ptr, len).unwrap()
        } else {
            std::ptr::null_mut()
        };

        (cpu_ptr, ptr, 0)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64) {
        let ptr = create_buffer::<T>(
            &self.ctx(),
            MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
            data.len(),
            Some(data),
        )
        .unwrap();

        #[cfg(not(feature = "safe"))]
        self.inner.borrow_mut().ptrs.push(ptr as *mut c_void);

        let cpu_ptr = if self.unified_mem() {
            unified_ptr::<T>(self.queue(), ptr, data.len()).unwrap()
        } else {
            std::ptr::null_mut()
        };

        (cpu_ptr, ptr, 0)
    }
}

impl<T> ManualMem<T> for CLDevice {
    fn drop_buf(&self, buf: crate::Buffer<T>) {
        unsafe {
            release_mem_object(buf.ptr.1).unwrap();
        }
    }
}

impl<T> CacheBuf<T> for CLDevice {
    fn cached_buf(&self, len: usize) -> Buffer<T> {
        CLCache::get::<T>(self, len)
    }
}

impl<T: CDatatype> ClearBuf<T> for CLDevice {
    fn clear(&self, buf: &mut Buffer<T>) {
        cl_clear(self, buf).unwrap()
    }
}

impl<T> WriteBuf<T> for CLDevice {
    fn write(&self, buf: &mut Buffer<T>, data: &[T]) {
        let event = unsafe { enqueue_write_buffer(&self.queue(), buf.ptr.1, data, false).unwrap() };
        wait_for_event(event).unwrap();
    }
}

impl<T: Default + Copy> VecRead<T> for CLDevice {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        // TODO: check null?
        assert!(
            !buf.ptr.1.is_null(),
            "called VecRead::read(..) on a non OpenCL buffer (this would read out a null pointer)"
        );
        let mut read = vec![T::default(); buf.len];
        let event =
            unsafe { enqueue_read_buffer(&self.queue(), buf.ptr.1, &mut read, false).unwrap() };
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
    pub fn new(device: CLIntDevice) -> crate::Result<InternCLDevice> {
        let ctx = create_context(&[device])?;
        let queue = create_command_queue(&ctx, device)?;
        let unified_mem = device.unified_mem()?;

        Ok(InternCLDevice {
            ptrs: Vec::new(),
            device,
            ctx,
            queue,
            unified_mem,
        })
    }
}

impl Drop for InternCLDevice {
    fn drop(&mut self) {
        unsafe {
            let count = get_device_count();
            *count -= 1;
            deallocate_cache(*count);
        }
    }
}

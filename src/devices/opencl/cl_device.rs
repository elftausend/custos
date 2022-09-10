use super::{
    api::{
        create_command_queue, create_context, enqueue_full_copy_buffer, enqueue_read_buffer,
        enqueue_write_buffer, unified_ptr, wait_for_event, CLIntDevice, CommandQueue, Context,
    },
    cl_clear, KernelCacheCL, RawCL, CL_DEVICES,
};
use crate::{
    cache::{Cache, CacheReturn},
    devices::opencl::api::{create_buffer, MemFlags},
    Alloc, AsDev, Buffer, CDatatype, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, Device, DeviceType,
    Error, Graph, GraphReturn, VecRead, WriteBuf, CPU, Device1,
};
use std::{
    cell::{Ref, RefCell},
    ffi::c_void,
    fmt::Debug,
};

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
pub struct OpenCL {
    pub kernel_cache: RefCell<KernelCacheCL>,
    pub cache: RefCell<Cache<RawCL>>,
    pub inner: RefCell<InternCLDevice>,
    pub graph: RefCell<Graph>,
    pub cpu: CPU,
}

unsafe impl Sync for InternCLDevice {}

impl OpenCL {
    /// Returns an [CLDevice] at the specified device index.
    /// # Errors
    /// - No device is found at the given device index
    /// - some other OpenCL related errors
    pub fn new(device_idx: usize) -> Result<OpenCL, Error> {
        let inner = RefCell::new(CL_DEVICES.current(device_idx)?);
        Ok(OpenCL {
            kernel_cache: RefCell::new(KernelCacheCL::default()),
            cache: RefCell::new(Cache::default()),
            inner,
            graph: RefCell::new(Graph::new()),
            cpu: CPU::new(),
        })
    }

    #[inline]
    pub fn ctx(&self) -> Ref<Context> {
        let borrow = self.inner.borrow();
        Ref::map(borrow, |device| &device.ctx)
    }

    #[inline]
    pub fn queue(&self) -> Ref<CommandQueue> {
        let borrow = self.inner.borrow();
        Ref::map(borrow, |device| &device.queue)
    }

    #[inline]
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

    #[inline]
    pub fn unified_mem(&self) -> bool {
        self.inner.borrow().unified_mem
    }

    pub fn set_unified_mem(&self, unified_mem: bool) {
        self.inner.borrow_mut().unified_mem = unified_mem;
    }
}

impl Debug for OpenCL {
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

impl Device1 for OpenCL {}

impl<T> Alloc<T> for OpenCL {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        let ptr =
            create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap();

        let cpu_ptr = if self.unified_mem() {
            // TODO: not unmapping before executing a kernel results in ub?
            unified_ptr::<T>(&self.queue(), ptr, len).unwrap()
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

        let cpu_ptr = if self.unified_mem() {
            unified_ptr::<T>(&self.queue(), ptr, data.len()).unwrap()
        } else {
            std::ptr::null_mut()
        };

        (cpu_ptr, ptr, 0)
    }

    fn as_dev(&self) -> Device {
        Device {
            device_type: DeviceType::CL,
            device: self as *const OpenCL as *mut u8,
        }
    }
}

impl<'a, T> CloneBuf<'a, T> for OpenCL {
    fn clone_buf(&'a self, buf: &Buffer<'a, T>) -> Buffer<'a, T> {
        let cloned = Buffer::new(self, buf.len);
        enqueue_full_copy_buffer::<T>(&self.queue(), buf.ptr.1, cloned.ptr.1, buf.len).unwrap();
        cloned
    }
}

impl<'a, T> CacheBuf<'a, T> for OpenCL {
    #[inline]
    fn cached(&'a self, len: usize) -> Buffer<'a, T> {
        Cache::get(self, len, CachedLeaf)
    }
}

impl CacheReturn<RawCL> for OpenCL {
    #[inline]
    fn cache(&self) -> std::cell::RefMut<Cache<RawCL>> {
        self.cache.borrow_mut()
    }
}

impl GraphReturn for OpenCL {
    #[inline]
    fn graph(&self) -> std::cell::RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

#[cfg(feature = "opt-cache")]
impl crate::GraphOpt for OpenCL {}

#[inline]
pub fn cl_cached<T>(device: &OpenCL, len: usize) -> Buffer<T> {
    device.cached(len)
}

impl<T: CDatatype> ClearBuf<T> for OpenCL {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T>) {
        cl_clear(self, buf).unwrap()
    }
}

impl<T> WriteBuf<T> for OpenCL {
    fn write(&self, buf: &mut Buffer<T>, data: &[T]) {
        let event = unsafe { enqueue_write_buffer(&self.queue(), buf.ptr.1, data, false).unwrap() };
        wait_for_event(event).unwrap();
    }
}

impl<T: Clone + Default> VecRead<T> for OpenCL {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
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

impl AsDev for OpenCL {}

/// Internal representation of an OpenCL Device with the capability of storing pointers.
/// # Note / Safety
///
/// If the 'safe' feature isn't used, all pointers will get invalid when the drop code for a CLDevice object is run as that deallocates the memory previously pointed at by the pointers stored in 'ptrs'.
#[derive(Debug, Clone)]
pub struct InternCLDevice {
    pub ptrs: Vec<*mut c_void>,
    device: CLIntDevice,
    ctx: Context,
    queue: CommandQueue,
    unified_mem: bool,
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

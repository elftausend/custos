use super::{
    api::{
        create_command_queue, create_context, enqueue_full_copy_buffer, enqueue_read_buffer,
        enqueue_write_buffer, wait_for_event, CLIntDevice, CommandQueue, Context,
    },
    cl_clear, CLPtr, KernelCacheCL, RawCL, CL_DEVICES,
};
use crate::{
    cache::{Cache, CacheReturn},
    devices::opencl::api::{create_buffer, MemFlags},
    Alloc, Buffer, CDatatype, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, Device, DevicelessAble,
    Error, Graph, GraphReturn, VecRead, WriteBuf, CPU,
};
use std::{
    cell::{Ref, RefCell},
    ffi::c_void,
    fmt::Debug,
};

#[cfg(unified_cl)]
use crate::{opencl::api::unified_ptr, CPUCL};

/// Used to perform calculations with an OpenCL capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CLDevice].
/// # Example
/// ```
/// use custos::{OpenCL, VecRead, Buffer, Error};
///
/// fn main() -> Result<(), Error> {
///     let device = OpenCL::new(0)?;
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

    /// Checks if the device supports unified memory.
    #[inline]
    pub fn unified_mem(&self) -> bool {
        self.inner.borrow().unified_mem
    }

    #[deprecated(since="0.6.0", note="Use the environment variable 'CUSTOS_USE_UNIFIED' set to 'true', 'false' or 'default'[=hardware dependent] instead.")]
    pub fn set_unified_mem(&self, unified_mem: bool) {
        self.inner.borrow_mut().unified_mem = unified_mem;
    }
}

impl Device for OpenCL {
    type P<U> = CLPtr<U>;
}

impl DevicelessAble for OpenCL {}

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

impl Alloc for OpenCL {
    fn alloc<T>(&self, len: usize) -> CLPtr<T> {
        let ptr =
            create_buffer::<T>(&self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap();

        #[cfg(unified_cl)]
        let host_ptr = unified_ptr::<T>(&self.queue(), ptr, len).unwrap();

        #[cfg(not(unified_cl))]
        let host_ptr = std::ptr::null_mut();

        CLPtr {
            ptr,
            host_ptr
        }
    }

    fn with_data<T>(&self, data: &[T]) -> CLPtr<T> {
        let ptr = create_buffer::<T>(
            &self.ctx(),
            MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
            data.len(),
            Some(data),
        )
        .unwrap();

        #[cfg(unified_cl)]
        let host_ptr = unified_ptr::<T>(&self.queue(), ptr, data.len()).unwrap();

        #[cfg(not(unified_cl))]
        let host_ptr = std::ptr::null_mut();

        CLPtr {
            ptr,
            host_ptr,
        }
    }
}

impl<'a, T> CloneBuf<'a, T> for OpenCL {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, OpenCL>) -> Buffer<'a, T, OpenCL> {
        let cloned = Buffer::new(self, buf.len);
        enqueue_full_copy_buffer::<T>(&self.queue(), buf.ptrs().1, cloned.ptrs().1, buf.len)
            .unwrap();
        cloned
    }
}

impl<'a, T> CacheBuf<'a, T> for OpenCL {
    #[inline]
    fn cached(&'a self, len: usize) -> Buffer<'a, T, OpenCL> {
        Cache::get(self, len, CachedLeaf)
    }
}

impl CacheReturn for OpenCL {
    type CT = RawCL;
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

#[cfg(unified_cl)]
impl CPUCL for OpenCL {}

#[cfg(feature = "opt-cache")]
impl crate::GraphOpt for OpenCL {}

#[inline]
pub fn cl_cached<T>(device: &OpenCL, len: usize) -> Buffer<T, OpenCL> {
    device.cached(len)
}

impl<T: CDatatype> ClearBuf<T, OpenCL> for OpenCL {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, OpenCL>) {
        cl_clear(self, buf).unwrap()
    }
}

impl<T> WriteBuf<T, OpenCL> for OpenCL {
    fn write(&self, buf: &mut Buffer<T, OpenCL>, data: &[T]) {
        let event =
            unsafe { enqueue_write_buffer(&self.queue(), buf.ptrs().1, data, true).unwrap() };
        wait_for_event(event).unwrap();
    }
}

impl<T: Clone + Default> VecRead<T, OpenCL> for OpenCL {
    fn read(&self, buf: &crate::Buffer<T, OpenCL>) -> Vec<T> {
        assert!(
            !buf.ptrs().1.is_null(),
            "called VecRead::read(..) on a non OpenCL buffer (this would read out a null pointer)"
        );
        let mut read = vec![T::default(); buf.len];
        let event =
            unsafe { enqueue_read_buffer(&self.queue(), buf.ptrs().1, &mut read, false).unwrap() };
        wait_for_event(event).unwrap();
        read
    }
}

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

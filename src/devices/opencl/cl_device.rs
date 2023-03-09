use min_cl::CLDevice;

use min_cl::api::{
    create_buffer, enqueue_full_copy_buffer, CLIntDevice, CommandQueue, Context, MemFlags,
};

use super::{chosen_cl_idx, enqueue_kernel, AsClCvoidPtr, CLPtr, KernelCacheCL, RawCL};
use crate::flag::AllocFlag;
use crate::{
    cache::{Cache, CacheReturn, RawConv},
    Alloc, Buffer, CloneBuf, Device, Error, Graph, GraphReturn, CPU,
};
use crate::{GlobalCount, Shape};

use std::{cell::RefCell, fmt::Debug};

#[cfg(unified_cl)]
use min_cl::api::unified_ptr;

/// Used to perform calculations with an OpenCL capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CLDevice].
/// # Example
/// ```
/// use custos::{OpenCL, Read, Buffer, Error};
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
    pub cache: RefCell<Cache<OpenCL>>,
    pub inner: CLDevice,
    pub graph: RefCell<Graph<GlobalCount>>,
    pub cpu: CPU,
    #[cfg(feature = "autograd")]
    pub tape: RefCell<crate::Tape<OpenCL>>,
}

/// Short form for `OpenCL`
pub type CL = OpenCL;

impl OpenCL {
    /// Returns an [OpenCL] at the specified device index.
    /// # Errors
    /// - No device is found at the given device index
    /// - some other OpenCL related errors
    pub fn new(device_idx: usize) -> Result<OpenCL, Error> {
        let inner = CLDevice::new(device_idx)?;
        Ok(OpenCL {
            inner,
            kernel_cache: Default::default(),
            cache: Default::default(),
            graph: Default::default(),
            cpu: Default::default(),
            #[cfg(feature = "autograd")]
            tape: Default::default(),
        })
    }

    /// Sets the values of the attributes cache, kernel cache, graph and CPU to their default.
    /// This cleans up any accumulated allocations.
    pub fn reset(&'static mut self) {
        self.kernel_cache = Default::default();
        self.cache = Default::default();
        self.graph = Default::default();
        self.cpu = Default::default();
    }

    #[inline]
    pub fn ctx(&self) -> &Context {
        &self.inner.ctx
    }

    #[inline]
    pub fn queue(&self) -> &CommandQueue {
        &self.inner.queue
    }

    #[inline]
    pub fn device(&self) -> CLIntDevice {
        self.inner.device
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

    /// Checks whether the device supports unified memory.
    #[inline]
    pub fn unified_mem(&self) -> bool {
        self.inner.unified_mem
    }

    #[deprecated(
        since = "0.6.0",
        note = "Use the environment variable 'CUSTOS_USE_UNIFIED' set to 'true', 'false' or 'default'[=hardware dependent] instead."
    )]
    pub fn set_unified_mem(&mut self, unified_mem: bool) {
        self.inner.unified_mem = unified_mem;
    }

    #[inline]
    pub fn launch_kernel(
        &self,
        src: &str,
        gws: [usize; 3],
        lws: Option<[usize; 3]>,
        args: &[&dyn AsClCvoidPtr],
    ) -> crate::Result<()> {
        enqueue_kernel(self, src, gws, lws, args)
    }
}

impl Default for OpenCL {
    fn default() -> Self {
        let inner = CLDevice::new(chosen_cl_idx()).expect("Could not get CLDevice.");
        Self {
            inner,
            kernel_cache: Default::default(),
            cache: Default::default(),
            graph: Default::default(),
            cpu: Default::default(),
            #[cfg(feature = "autograd")]
            tape: Default::default(),
        }
    }
}

impl Device for OpenCL {
    type Ptr<U, S: Shape> = CLPtr<U>;
    type Cache = Cache<Self>;

    fn new() -> crate::Result<Self> {
        OpenCL::new(chosen_cl_idx())
    }
}

#[cfg(feature = "autograd")]
impl crate::TapeReturn for OpenCL {
    #[inline]
    fn tape(&self) -> core::cell::Ref<crate::Tape<Self>> {
        self.tape.borrow()
    }

    #[inline]
    fn tape_mut(&self) -> core::cell::RefMut<crate::Tape<Self>> {
        self.tape.borrow_mut()
    }
}

impl RawConv for OpenCL {
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, flag: AllocFlag) -> Self::CT {
        RawCL {
            ptr: ptr.ptr,
            host_ptr: ptr.host_ptr as *mut u8,
            len,
            flag,
        }
    }

    fn destruct<T, S: Shape>(ct: &Self::CT) -> Self::Ptr<T, S> {
        CLPtr {
            ptr: ct.ptr,
            host_ptr: ct.host_ptr as *mut T,
            len: ct.len,
            flag: ct.flag,
        }
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

impl<T, S: Shape> Alloc<'_, T, S> for OpenCL {
    fn alloc(&self, mut len: usize, flag: AllocFlag) -> CLPtr<T> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        let ptr = create_buffer::<T>(self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap();

        #[cfg(unified_cl)]
        let host_ptr = unified_ptr::<T>(self.queue(), ptr, len).unwrap();

        #[cfg(not(unified_cl))]
        let host_ptr = std::ptr::null_mut();

        CLPtr {
            ptr,
            host_ptr,
            len,
            flag,
        }
    }

    fn with_slice(&self, data: &[T]) -> CLPtr<T> {
        let ptr = create_buffer::<T>(
            self.ctx(),
            MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
            data.len(),
            Some(data),
        )
        .unwrap();

        #[cfg(unified_cl)]
        let host_ptr = unified_ptr::<T>(self.queue(), ptr, data.len()).unwrap();

        #[cfg(not(unified_cl))]
        let host_ptr = std::ptr::null_mut();

        CLPtr {
            ptr,
            host_ptr,
            len: data.len(),
            flag: AllocFlag::None,
        }
    }
}

impl<'a, T> CloneBuf<'a, T> for OpenCL {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, OpenCL>) -> Buffer<'a, T, OpenCL> {
        let cloned = Buffer::new(self, buf.len());
        enqueue_full_copy_buffer::<T>(self.queue(), buf.ptr.ptr, cloned.ptr.ptr, buf.len())
            .unwrap();
        cloned
    }
}

impl CacheReturn for OpenCL {
    type CT = RawCL;
    #[inline]
    fn cache(&self) -> std::cell::RefMut<Cache<OpenCL>>
    where
        OpenCL: RawConv,
    {
        self.cache.borrow_mut()
    }
}

impl GraphReturn for OpenCL {
    #[inline]
    fn graph(&self) -> std::cell::RefMut<Graph<GlobalCount>> {
        self.graph.borrow_mut()
    }
}

#[cfg(unified_cl)]
impl crate::MainMemory for OpenCL {
    #[inline]
    fn as_ptr<T, S: Shape>(ptr: &Self::Ptr<T, S>) -> *const T {
        ptr.host_ptr
    }

    #[inline]
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Ptr<T, S>) -> *mut T {
        ptr.host_ptr
    }
}

#[cfg(test)]
mod tests {
    use crate::{opencl::cl_device::CLDevice, Buffer, OpenCL};

    #[test]
    fn test_multiplie_queues() -> crate::Result<()> {
        let device = CLDevice::new(0)?;
        let cl = OpenCL {
            kernel_cache: Default::default(),
            cache: Default::default(),
            inner: device,
            graph: Default::default(),
            cpu: Default::default(),
            #[cfg(feature = "autograd")]
            tape: Default::default(),
        };

        let buf = Buffer::from((&cl, &[1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(buf.read(), vec![1, 2, 3, 4, 5, 6, 7]);

        let device = CLDevice::new(0)?;

        let cl1 = OpenCL {
            kernel_cache: Default::default(),
            cache: Default::default(),
            inner: device,
            graph: Default::default(),
            cpu: Default::default(),
            #[cfg(feature = "autograd")]
            tape: Default::default(),
        };

        let buf = Buffer::from((&cl1, &[2, 2, 4, 4, 2, 1, 3]));
        assert_eq!(buf.read(), vec![2, 2, 4, 4, 2, 1, 3]);

        Ok(())
    }
}

use min_cl::CLDevice;

use min_cl::api::{
    create_buffer, enqueue_full_copy_buffer, enqueue_read_buffer, enqueue_write_buffer,
    wait_for_event, CLIntDevice, CommandQueue, Context, MemFlags,
};

use super::{chosen_cl_idx, cl_clear, CLPtr, KernelCacheCL, RawCL};
use crate::flag::AllocFlag;
use crate::{Shape, Cache2, CacheReturn2, Node, BufType};
use crate::{
    cache::{Cache, CacheReturn, RawConv},
    Alloc, Buffer, CDatatype, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, Device, Error, Graph,
    GraphReturn, Read, WriteBuf, CPU,
};
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
    pub cache: RefCell<Cache2<OpenCL>>,
    pub inner: CLDevice,
    pub graph: RefCell<Graph>,
    pub cpu: CPU,
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
}

impl Device for OpenCL {
    type Ptr<U, S: Shape> = CLPtr<U>;
    type Cache = Cache2<Self>;

    fn new() -> crate::Result<Self> {
        OpenCL::new(chosen_cl_idx())
    }
}

impl BufType for crate::OpenCL {
    type Deallocator = RawCL;

    unsafe fn ptr_to_raw<T, S: Shape>(ptr: &Self::Ptr<u8, S>) -> Self::Deallocator {
        RawCL {
            ptr: ptr.ptr,
            host_ptr: ptr.host_ptr as *mut u8,
            len: ptr.len,
            // FIXME: mind default node
            node: Node::default(),
        }
    }
}
impl RawConv for OpenCL {
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, node: crate::Node) -> Self::CT {
        RawCL {
            ptr: ptr.ptr,
            host_ptr: ptr.host_ptr as *mut u8,
            len,
            node,
        }
    }

    fn destruct<T, S: Shape>(ct: &Self::CT, flag: AllocFlag) -> (Self::Ptr<T, S>, crate::Node) {
        (
            CLPtr {
                ptr: ct.ptr,
                host_ptr: ct.host_ptr as *mut T,
                len: ct.len,
                flag,
            },
            ct.node,
        )
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
    unsafe fn alloc<A>(&self, mut len: usize, flag: AllocFlag) -> CLPtr<T> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        CLPtr::<T>::new::<A>(&self.inner, len, flag)
    }

    fn with_slice(&self, data: &[T]) -> CLPtr<T> {
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
            len: data.len(),
            flag: AllocFlag::None,
        }
    }
}

impl<'a, T> CloneBuf<'a, T> for OpenCL {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, OpenCL>) -> Buffer<'a, T, OpenCL> {
        let cloned = Buffer::new(self, buf.len());
        enqueue_full_copy_buffer::<T>(&self.queue(), buf.ptr.ptr, cloned.ptr.ptr, buf.len())
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
    fn cache(&self) -> std::cell::RefMut<Cache<OpenCL>>
    where
        OpenCL: RawConv,
    {
        todo!()
    }
}

impl CacheReturn2 for OpenCL {
    #[inline]
    fn cache(&self) -> std::cell::RefMut<Cache2<OpenCL>> {
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
            unsafe { enqueue_write_buffer(&self.queue(), buf.cl_ptr(), data, true).unwrap() };
        wait_for_event(event).unwrap();
    }
}

/*#[cfg(not(unified_cl))]
impl<T: Clone + Default> Read<T, OpenCL> for OpenCL {
    type Read<'a> = Vec<T> where T: 'a;

    fn read<'a>(&self, buf: &'a Buffer<T, OpenCL>) -> Self::Read<'a> {
        self.read_to_vec(buf)
    }

    #[inline]
    fn read_to_vec(&self, buf: &crate::Buffer<T, OpenCL>) -> Vec<T> {
        read_cl_buf_to_vec(self, buf).unwrap()
    }
}*/

impl<T: Clone + Default> Read<T, OpenCL> for OpenCL {
    #[cfg(not(unified_cl))]
    type Read<'a> = Vec<T> where T: 'a;
    #[cfg(unified_cl)]
    type Read<'a> = &'a [T] where T: 'a;

    #[cfg(not(unified_cl))]
    fn read<'a>(&self, buf: &'a Buffer<T, OpenCL>) -> Self::Read<'a> {
        self.read_to_vec(buf)
    }

    #[cfg(unified_cl)]
    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, OpenCL>) -> Self::Read<'a> {
        buf.as_slice()
    }

    #[inline]
    fn read_to_vec(&self, buf: &Buffer<T, OpenCL>) -> Vec<T> {
        read_cl_buf_to_vec(self, buf).unwrap()
    }
}

fn read_cl_buf_to_vec<T: Clone + Default>(
    device: &OpenCL,
    buf: &Buffer<T, OpenCL>,
) -> crate::Result<Vec<T>> {
    let mut read = vec![T::default(); buf.len()];
    let event = unsafe { enqueue_read_buffer(&device.queue(), buf.cl_ptr(), &mut read, false)? };
    wait_for_event(event).unwrap();
    Ok(read)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

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
        };

        let buf = Buffer::from((&cl1, &[2, 2, 4, 4, 2, 1, 3]));
        assert_eq!(buf.read(), vec![2, 2, 4, 4, 2, 1, 3]);

        Ok(())
    }
}

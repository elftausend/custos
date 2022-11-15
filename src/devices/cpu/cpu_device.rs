use crate::{
    devices::cache::{Cache, CacheReturn},
    Alloc, Buffer, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, Device, DevicelessAble, Graph,
    GraphReturn, Read, WriteBuf, CPUCL,
};
use alloc::{
    alloc::{handle_alloc_error, Layout},
    vec::Vec,
};
use core::{
    cell::{RefCell, RefMut},
    fmt::Debug,
    mem::size_of,
};

use super::{CPUPtr, RawCpuBuf};

#[derive(Debug, Default)]
/// A CPU is used to perform calculations on the host CPU.
/// To make new operations invocable, a trait providing new functions should be implemented for [CPU].
///
/// # Example
/// ```
/// use custos::{CPU, Read, Buffer};
///
/// let device = CPU::new();
/// let a = Buffer::from((&device, [1, 2, 3]));
///
/// let out = device.read(&a);
///
/// assert_eq!(out, vec![1, 2, 3]);
/// ```
pub struct CPU {
    pub cache: RefCell<Cache<RawCpuBuf>>,
    pub graph: RefCell<Graph>,
}

impl CPU {
    /// Creates an [CPU] with an InternCPU that holds an empty vector of pointers.
    #[must_use]
    pub fn new() -> CPU {
        CPU {
            cache: RefCell::new(Cache::default()),
            graph: RefCell::new(Graph::new()),
        }
    }
}

impl Device for CPU {
    type Ptr<U, const N: usize> = CPUPtr<U>;
    type Cache<const N: usize> = Cache<RawCpuBuf>;

    fn new() -> crate::Result<Self> {
        Ok(Self::new())
    }
}

impl<T> DevicelessAble<T> for CPU {}

impl<T> Alloc<T> for CPU {
    fn alloc(&self, len: usize) -> CPUPtr<T> {
        assert!(len > 0, "invalid buffer len: 0");
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = unsafe { alloc::alloc::alloc(layout) };

        // initialize block of memory
        for element in unsafe { alloc::slice::from_raw_parts_mut(ptr, len * size_of::<T>()) } {
            *element = 0;
        }

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        CPUPtr { ptr: ptr as *mut T }
    }

    fn with_slice(&self, data: &[T]) -> CPUPtr<T>
    where
        T: Clone,
    {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let cpu_ptr = self.alloc(data.len());
        let slice = unsafe { alloc::slice::from_raw_parts_mut(cpu_ptr.ptr, data.len()) };
        slice.clone_from_slice(data);

        cpu_ptr
    }
    fn alloc_with_vec(&self, mut vec: Vec<T>) -> CPUPtr<T> {
        assert!(!vec.is_empty(), "invalid buffer len: 0");

        let ptr = vec.as_mut_ptr();
        core::mem::forget(vec);

        CPUPtr { ptr }
    }
}

impl CacheReturn for CPU {
    type CT = RawCpuBuf;
    #[inline]
    fn cache(&self) -> RefMut<Cache<RawCpuBuf>> {
        self.cache.borrow_mut()
    }
}

impl GraphReturn for CPU {
    #[inline]
    fn graph(&self) -> RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

impl CPUCL for CPU {
    #[inline]
    fn buf_as_slice<'a, T, const N: usize>(buf: &'a Buffer<T, Self, N>) -> &'a [T] {
        assert!(
            !buf.ptrs().0.is_null(),
            "called as_slice() on an invalid CPU buffer (this would dereference an invalid pointer)"
        );
        unsafe { alloc::slice::from_raw_parts(buf.ptrs().0, buf.len) }
    }

    fn buf_as_slice_mut<'a, T, const N: usize>(buf: &'a mut Buffer<T, Self, N>) -> &'a mut [T] {
        assert!(
            !buf.ptrs().0.is_null(),
            "called as_slice() on an invalid CPU buffer (this would dereference an invalid pointer)"
        );
        unsafe { alloc::slice::from_raw_parts_mut(buf.ptrs_mut().0, buf.len) }
    }
}

#[cfg(feature = "opt-cache")]
impl crate::GraphOpt for CPU {}

impl<'a, T: Clone> CloneBuf<'a, T> for CPU {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU>) -> Buffer<'a, T, CPU> {
        let mut cloned = Buffer::<_, _, 0>::new(self, buf.len);
        cloned.clone_from_slice(buf);
        cloned
    }
}

impl<'a, T> CacheBuf<'a, T> for CPU {
    #[inline]
    fn cached(&'a self, len: usize) -> Buffer<'a, T, CPU> {
        Cache::get::<T, CPU, 0>(self, len, CachedLeaf)
    }
}

#[inline]
pub fn cpu_cached<T: Clone>(device: &CPU, len: usize) -> Buffer<T, CPU> {
    device.cached(len)
}

impl<T, D: CPUCL> Read<T, D> for CPU {
    type Read<'a> = &'a [T] where T: 'a, D: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, D>) -> Self::Read<'a> {
        buf.as_slice()
    }

    #[inline]
    fn read_to_vec<'a>(&self, buf: &Buffer<T, D>) -> Vec<T>
    where
        T: Default + Clone,
    {
        buf.to_vec()
    }
}

impl<T: Default, D: CPUCL> ClearBuf<T, D> for CPU {
    fn clear(&self, buf: &mut Buffer<T, D>) {
        for value in buf {
            *value = T::default();
        }
    }
}

impl<T: Copy, D: CPUCL> WriteBuf<T, D> for CPU {
    fn write(&self, buf: &mut Buffer<T, D>, data: &[T]) {
        buf.copy_from_slice(data)
    }
}

use crate::{
    devices::cache::{Cache, CacheReturn},
    Alloc, Buffer, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, DevicelessAble, Graph, GraphReturn,
    VecRead, WriteBuf, CPUCL, Device,
};
use std::{
    alloc::{handle_alloc_error, Layout},
    cell::{RefCell, RefMut},
    ffi::c_void,
    fmt::Debug,
    mem::size_of,
};

use super::{RawCpuBuf, CPUPtr};

#[derive(Debug, Default)]
/// A CPU is used to perform calculations on the host CPU.
/// To make new operations invocable, a trait providing new functions should be implemented for [CPU].
///
/// # Example
/// ```
/// use custos::{CPU, VecRead, Buffer};
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
    type P<U> = CPUPtr<U>;
}
impl DevicelessAble for CPU {}

impl Alloc for CPU {
    fn alloc<T>(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        assert!(len > 0, "invalid buffer len: 0");
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };

        // initialize block of memory
        for element in unsafe { std::slice::from_raw_parts_mut(ptr, len * size_of::<T>()) } {
            *element = 0;
        }

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        (ptr as *mut T, std::ptr::null_mut(), 0)
    }

    fn with_data<T>(&self, data: &[T]) -> (*mut T, *mut c_void, u64)
    where
        T: Clone,
    {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let (ptr, _, _) = self.alloc(data.len());
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) };
        slice.clone_from_slice(data);
        (ptr, std::ptr::null_mut(), 0)
    }
    fn alloc_with_vec<T>(&self, mut vec: Vec<T>) -> (*mut T, *mut c_void, u64) {
        assert!(!vec.is_empty(), "invalid buffer len: 0");

        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);

        (ptr, std::ptr::null_mut(), 0)
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

impl CPUCL for CPU {}

#[cfg(feature = "opt-cache")]
impl crate::GraphOpt for CPU {}

impl<'a, T: Clone> CloneBuf<'a, T> for CPU {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU>) -> Buffer<'a, T, CPU> {
        let mut cloned = Buffer::new(self, buf.len);
        cloned.clone_from_slice(buf);
        cloned
    }
}

impl<'a, T> CacheBuf<'a, T> for CPU {
    #[inline]
    fn cached(&'a self, len: usize) -> Buffer<'a, T, CPU> {
        Cache::get::<T, CPU>(self, len, CachedLeaf)
    }
}

#[inline]
pub fn cpu_cached<T: Clone>(device: &CPU, len: usize) -> Buffer<T, CPU> {
    device.cached(len)
}

impl<T: Clone, D: CPUCL> VecRead<T, D> for CPU {
    fn read(&self, buf: &Buffer<T, D>) -> Vec<T> {
        buf.as_slice().to_vec()
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

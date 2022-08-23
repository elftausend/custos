use crate::{
    devices::cache::{Cache, CacheReturn},
    Alloc, AsDev, Buffer, CacheBuf, ClearBuf, CloneBuf, Device, DeviceType, Graph, GraphReturn,
    VecRead, WriteBuf,
};
use std::{
    alloc::{handle_alloc_error, Layout},
    cell::{RefCell, RefMut},
    ffi::c_void,
    fmt::Debug,
    mem::size_of,
};

use super::RawCpuBuf;

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

impl<T> Alloc<T> for CPU {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
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

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64)
    where
        T: Clone,
    {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let (ptr, _, _) = self.alloc(data.len());
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) };
        slice.clone_from_slice(data);
        (ptr, std::ptr::null_mut(), 0)
    }
    fn alloc_with_vec(&self, mut vec: Vec<T>) -> (*mut T, *mut c_void, u64) {
        assert!(!vec.is_empty(), "invalid buffer len: 0");

        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);

        (ptr, std::ptr::null_mut(), 0)
    }

    fn as_dev(&self) -> crate::Device {
        Device {
            device_type: DeviceType::CPU,
            device: self as *const CPU as *mut u8,
        }
    }
}

impl AsDev for CPU {}

impl CacheReturn<RawCpuBuf> for CPU {
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

impl<'a, T: Clone> CloneBuf<'a, T> for CPU {
    fn clone_buf(&'a self, buf: &Buffer<'a, T>) -> Buffer<'a, T> {
        let mut cloned = Buffer::new(self, buf.len);
        cloned.clone_from_slice(buf);
        cloned
    }
}

impl<'a, T> CacheBuf<'a, T> for CPU {
    #[inline]
    fn cached(&'a self, len: usize) -> Buffer<'a, T> {
        let node = self.graph().add_leaf(len);
        Cache::get::<T, CPU, _>(self, len, node.idx)
    }
}

#[inline]
pub fn cpu_cached<T: Clone>(device: &CPU, len: usize) -> Buffer<T> {
    device.cached(len)
}

impl<T: Clone> VecRead<T> for CPU {
    fn read(&self, buf: &Buffer<T>) -> Vec<T> {
        buf.as_slice().to_vec()
    }
}

impl<T: Default> ClearBuf<T> for CPU {
    fn clear(&self, buf: &mut Buffer<T>) {
        for value in buf {
            *value = T::default();
        }
    }
}

impl<T: Copy> WriteBuf<T> for CPU {
    fn write(&self, buf: &mut Buffer<T>, data: &[T]) {
        buf.copy_from_slice(data)
    }
}

use crate::{
    cache::RawConv,
    devices::cache::{Cache, CacheReturn},
    flag::AllocFlag,
    shape::Shape,
    Alloc, Buffer, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, Device, DevicelessAble, Graph,
    GraphReturn, MainMemory, Read, WriteBuf,
};
use core::{
    cell::{RefCell, RefMut},
    fmt::Debug,
    mem::{align_of, size_of},
};
use std::vec::Vec;

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
    pub cache: RefCell<Cache<CPU>>,
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
    type Ptr<U, S: Shape> = CPUPtr<U>;
    type Cache = Cache<CPU>; //<CPU as CacheReturn>::CT

    fn new() -> crate::Result<Self> {
        Ok(Self::new())
    }
}

impl RawConv for CPU {
    #[inline]
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, node: crate::Node) -> Self::CT {
        RawCpuBuf {
            ptr: ptr.ptr.cast(),
            len,
            align: align_of::<T>(),
            size: size_of::<T>(),
            node,
        }
    }

    #[inline]
    fn destruct<T, S: Shape>(ct: &Self::CT) -> (Self::Ptr<T, S>, crate::Node) {
        (
            CPUPtr {
                ptr: ct.ptr as *mut T,
                len: ct.len,
                flag: AllocFlag::Cache,
            },
            ct.node,
        )
    }
}

impl<'a, T> DevicelessAble<'a, T> for CPU {}

impl<T, S: Shape> Alloc<'_, T, S> for CPU {
    fn alloc(&self, mut len: usize, flag: AllocFlag) -> CPUPtr<T> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        CPUPtr::new(len, flag)
    }

    fn with_slice(&self, data: &[T]) -> CPUPtr<T>
    where
        T: Clone,
    {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let cpu_ptr = Alloc::<T>::alloc(self, data.len(), AllocFlag::None);
        //= self.alloc(data.len());
        let slice = unsafe { std::slice::from_raw_parts_mut(cpu_ptr.ptr, data.len()) };
        slice.clone_from_slice(data);

        cpu_ptr
    }
    fn alloc_with_vec(&self, mut vec: Vec<T>) -> CPUPtr<T> {
        assert!(!vec.is_empty(), "invalid buffer len: 0");

        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        core::mem::forget(vec);

        CPUPtr {
            ptr,
            len,
            flag: AllocFlag::None,
        }
    }
}

impl CacheReturn for CPU {
    type CT = RawCpuBuf;
    #[inline]
    fn cache(&self) -> RefMut<Cache<CPU>> {
        self.cache.borrow_mut()
    }
}

impl GraphReturn for CPU {
    #[inline]
    fn graph(&self) -> RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

impl MainMemory for CPU {
    #[inline(always)]
    fn buf_as_slice<'a, T, S: Shape>(buf: &'a Buffer<T, Self, S>) -> &'a [T] {
        assert!(
            !buf.ptr.ptr.is_null(),
            "called as_slice() on an invalid CPU buffer (this would dereference an invalid pointer)"
        );
        unsafe { std::slice::from_raw_parts(buf.ptr.ptr, buf.len()) }
    }

    #[inline(always)]
    fn buf_as_slice_mut<'a, T, S: Shape>(buf: &'a mut Buffer<T, Self, S>) -> &'a mut [T] {
        assert!(
            !buf.ptr.ptr.is_null(),
            "called as_slice() on an invalid CPU buffer (this would dereference an invalid pointer)"
        );
        unsafe { std::slice::from_raw_parts_mut(buf.ptr.ptr, buf.len()) }
    }
}

#[cfg(feature = "opt-cache")]
impl crate::GraphOpt for CPU {}

impl<'a, T: Clone, S: Shape> CloneBuf<'a, T, S> for CPU {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU, S>) -> Buffer<'a, T, CPU, S> {
        let mut cloned = Buffer::new(self, buf.len());
        cloned.clone_from_slice(buf);
        cloned
    }
}

impl<'a, T> CacheBuf<'a, T> for CPU {
    #[inline]
    fn cached(&'a self, len: usize) -> Buffer<'a, T, CPU> {
        Cache::get::<T, ()>(self, len, CachedLeaf)
    }
}

#[inline]
pub fn cpu_cached<T: Clone>(device: &CPU, len: usize) -> Buffer<T, CPU> {
    device.cached(len)
}

impl<T, D: MainMemory, S: Shape> Read<T, D, S> for CPU {
    type Read<'a> = &'a [T] where T: 'a, D: 'a, S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, D, S>) -> Self::Read<'a> {
        buf.as_slice()
    }

    #[inline]
    fn read_to_vec<'a>(&self, buf: &Buffer<T, D, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        buf.to_vec()
    }
}

impl<T: Default, D: MainMemory> ClearBuf<T, D> for CPU {
    fn clear(&self, buf: &mut Buffer<T, D>) {
        for value in buf {
            *value = T::default();
        }
    }
}

impl<T: Copy, D: MainMemory> WriteBuf<T, D> for CPU {
    fn write(&self, buf: &mut Buffer<T, D>, data: &[T]) {
        buf.copy_from_slice(data)
    }
}

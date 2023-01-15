use crate::{
    cache::RawConv,
    devices::cache::{Cache, CacheReturn},
    flag::AllocFlag,
    shape::Shape,
    Alloc, BufType, Buffer, Cache2, CacheBuf, CachedLeaf, ClearBuf, CloneBuf, Device,
    DevicelessAble, Graph, GraphReturn, MainMemory, Read, WriteBuf, CacheReturn2,
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
pub struct CPU<'a> {
    pub cache: RefCell<Cache2<'a, CPU<'a>>>,
    pub graph: RefCell<Graph>,
}

impl<'a> CPU<'a> {
    /// Creates an [CPU] with an InternCPU that holds an empty vector of pointers.
    #[must_use]
    pub fn new() -> Self {
        CPU {
            cache: RefCell::new(Cache2::default()),
            graph: RefCell::new(Graph::new()),
        }
    }
}

impl<'a> Device for CPU<'a> {
    type Ptr<U, S: Shape> = CPUPtr<U>;
    type Cache<'b> = Cache2<'a, CPU<'a>> where Self: 'b; //<CPU as CacheReturn>::CT
    //type Cache = ();

    fn new() -> crate::Result<Self> {
        Ok(Self::new())
    }
}

impl<'a> BufType for crate::CPU<'a> {
    type Deallocator = RawCpuBuf;

    unsafe fn ptr_to_raw<T, S: Shape>(ptr: &Self::Ptr<u8, S>) -> Self::Deallocator {
        RawCpuBuf {
            ptr: ptr.ptr,
            len: ptr.len,
            align: align_of::<T>(),
            size: size_of::<T>(),
            // FIXME: mind default node
            node: crate::Node::default(),
        }
    }
}

impl<'a> RawConv for CPU<'a> {
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
    fn destruct<T, S: Shape>(ct: &Self::CT, flag: AllocFlag) -> (Self::Ptr<T, S>, crate::Node) {
        (
            CPUPtr {
                ptr: ct.ptr as *mut T,
                len: ct.len,
                flag,
            },
            ct.node,
        )
    }
}

impl<'a, T> DevicelessAble<'a, T> for CPU<'a> {}

impl<'a, T, S: Shape> Alloc<'a, T, S> for CPU<'a> {
    unsafe fn alloc<A>(&self, mut len: usize, flag: AllocFlag) -> CPUPtr<T> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        CPUPtr::<T>::new::<A>(len, flag)
    }

    fn with_slice(&self, data: &[T]) -> CPUPtr<T>
    where
        T: Clone,
    {
        todo!()
        /*assert!(!data.is_empty(), "invalid buffer len: 0");
        let cpu_ptr = unsafe { Alloc::<T>::alloc::<T>(self, data.len(), AllocFlag::None) };
        //= self.alloc(data.len());
        let slice = unsafe { std::slice::from_raw_parts_mut(cpu_ptr.ptr, data.len()) };
        slice.clone_from_slice(data);

        cpu_ptr*/
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

impl<'a> CacheReturn for CPU<'a> {
    type CT = RawCpuBuf;
    #[inline]
    fn cache(&self) -> RefMut<Cache<CPU<'a>>> {
        todo!()
 //       self.cache.borrow_mut()
    }
}

impl<'a> CacheReturn2<'a> for CPU<'a> {
    #[inline]
    fn cache(&'a self) -> RefMut<Cache2<CPU<'a>>> {
        self.cache.borrow_mut()
    }
}

impl<'a> GraphReturn for CPU<'a> {
    #[inline]
    fn graph(&self) -> RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

impl<'a> MainMemory for CPU<'a> {
    #[inline]
    fn as_ptr<T, S: Shape>(ptr: &Self::Ptr<T, S>) -> *const T {
        ptr.ptr
    }

    #[inline]
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Ptr<T, S>) -> *mut T {
        ptr.ptr
    }
}

#[cfg(feature = "opt-cache")]
impl crate::GraphOpt for CPU {}

impl<'a, T: Clone, S: Shape> CloneBuf<'a, T, S> for CPU<'a> {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU, S>) -> Buffer<'a, T, CPU, S> {
        let mut cloned = Buffer::new(self, buf.len());
        cloned.clone_from_slice(buf);
        cloned
    }
}

impl<'a, T> CacheBuf<'a, T> for CPU<'a> {
    #[inline]
    fn cached(&'a self, len: usize) -> Buffer<'a, T, CPU<'a>> {
        Cache::get::<T, ()>(self, len, CachedLeaf)
    }
}

#[inline]
pub fn cpu_cached<'a, T: Clone>(device: &'a CPU<'a>, len: usize) -> Buffer<'a, T, CPU<'a>> {
    device.cached(len)
}

impl<T, D: MainMemory, S: Shape> Read<T, D, S> for CPU<'_> {
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

impl<T: Default, D: MainMemory> ClearBuf<T, D> for CPU<'_> {
    fn clear(&self, buf: &mut Buffer<T, D>) {
        for value in buf {
            *value = T::default();
        }
    }
}

impl<T: Copy, D: MainMemory> WriteBuf<T, D> for CPU<'_> {
    fn write(&self, buf: &mut Buffer<T, D>, data: &[T]) {
        buf.copy_from_slice(data)
    }
}

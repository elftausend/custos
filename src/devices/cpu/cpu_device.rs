use crate::{
    cache::RawConv,
    devices::cache::{Cache, CacheReturn},
    flag::AllocFlag,
    shape::Shape,
    Alloc, Buffer, CloneBuf, Device, DevicelessAble, Graph, GraphReturn, MainMemory,
};

use core::{
    cell::{RefCell, RefMut},
    fmt::Debug,
    mem::{align_of, size_of},
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
    pub cache: RefCell<Cache<CPU>>,
    pub graph: RefCell<Graph>,
    #[cfg(feature = "autograd")]
    pub tape: RefCell<crate::Tape<CPU>>,
}

impl CPU {
    /// Creates an [CPU] with an InternCPU that holds an empty vector of pointers.
    #[must_use]
    pub fn new() -> CPU {
        CPU {
            cache: Default::default(),
            graph: Default::default(),
            #[cfg(feature = "autograd")]
            tape: Default::default(),
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
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, flag: AllocFlag) -> Self::CT {
        RawCpuBuf {
            flag,
            len,
            ptr: ptr.ptr.cast(),
            align: align_of::<T>(),
            size: size_of::<T>(),
        }
    }

    #[inline]
    fn destruct<T, S: Shape>(ct: &Self::CT) -> Self::Ptr<T, S> {
        CPUPtr {
            ptr: ct.ptr as *mut T,
            len: ct.len,
            flag: ct.flag,
        }
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

#[cfg(feature = "autograd")]
impl crate::TapeReturn for CPU {
    #[inline]
    fn tape_mut(&self) -> RefMut<crate::Tape<Self>> {
        self.tape.borrow_mut()
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

impl<'a, T: Clone, S: Shape> CloneBuf<'a, T, S> for CPU {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU, S>) -> Buffer<'a, T, CPU, S> {
        let mut cloned = Buffer::new(self, buf.len());
        cloned.clone_from_slice(buf);
        cloned
    }
}

use crate::{
    deallocate_cache, get_device_count, libs::cpu::CPUCache, number::Number, AsDev, BaseDevice,
    Buffer, CDatatype, CacheBuf, ClearBuf, Device, GenericBlas, ManualMem, VecRead, WriteBuf,
};
use std::{cell::RefCell, ffi::c_void, fmt::Debug, rc::Rc};

#[derive(Debug, Clone, Default)]
/// A CPU is used to perform calculations on the host CPU.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CPU].
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
    pub inner: Rc<RefCell<InternCPU>>,
}

impl CPU {
    #[must_use]
    /// Creates an [CPU] with an InternCPU that holds an empty vector of pointers.
    pub fn new() -> CPU {
        unsafe {
            *get_device_count() += 1;
        }

        CPU {
            inner: Rc::new(RefCell::new(InternCPU { ptrs: Vec::new() })),
        }
    }
}

impl From<Rc<RefCell<InternCPU>>> for CPU {
    fn from(inner: Rc<RefCell<InternCPU>>) -> Self {
        CPU { inner }
    }
}

impl<T: Clone + Default> Device<T> for CPU {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        assert!(len > 0, "invalid buffer len: 0");
        let ptr = Box::into_raw(vec![T::default(); len].into_boxed_slice());
        //let size = std::mem::size_of::<T>() * len;

        #[cfg(not(feature = "safe"))]
        self.inner.borrow_mut().ptrs.push(StoredCPUPtr::new(
            ptr as *mut [u8],
            // TODO: use align of?
            std::mem::size_of::<T>(),
        ));
        (ptr as *mut T, std::ptr::null_mut(), 0)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64) {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let ptr = Box::into_raw(data.to_vec().into_boxed_slice());

        #[cfg(not(feature = "safe"))]
        self.inner.borrow_mut().ptrs.push(StoredCPUPtr::new(
            ptr as *mut [u8],
            std::mem::size_of::<T>(),
        ));
        (ptr as *mut T, std::ptr::null_mut(), 0)
    }
    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void, u64) {
        assert!(!vec.is_empty(), "invalid buffer len: 0");
        let ptr = Box::into_raw(vec.into_boxed_slice());

        #[cfg(not(feature = "safe"))]
        self.inner.borrow_mut().ptrs.push(StoredCPUPtr::new(
            ptr as *mut [u8],
            std::mem::size_of::<T>(),
        ));
        (ptr as *mut T, std::ptr::null_mut(), 0)
    }
}

impl AsDev for CPU {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(None, Some(Rc::downgrade(&self.inner)), None)
    }
}

impl<T> ManualMem<T> for CPU {
    fn drop_buf(&self, buf: Buffer<T>) {
        unsafe {
            Box::from_raw(buf.ptr.0);
        }
    }
}

impl<T: Copy + Default> CacheBuf<T> for CPU {
    fn cached_buf(&self, len: usize) -> Buffer<T> {
        CPUCache::get::<T>(self, len)
    }
}

impl<T: Copy + Default> VecRead<T> for CPU {
    fn read(&self, buf: &Buffer<T>) -> Vec<T> {
        unsafe { std::slice::from_raw_parts(buf.ptr.0, buf.len).to_vec() }
    }
}

impl<T: Number> ClearBuf<T> for CPU {
    fn clear(&self, buf: &mut Buffer<T>) {
        for value in buf {
            *value = T::zero();
        }
    }
}

impl<T: Copy> WriteBuf<T> for CPU {
    fn write(&self, buf: &mut Buffer<T>, data: &[T]) {
        buf.copy_from_slice(data)
    }
}

impl<T: CDatatype + GenericBlas> BaseDevice<T> for CPU {}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StoredCPUPtr {
    fat_ptr: *mut [u8],
    align: usize,
}

impl StoredCPUPtr {
    pub fn new(fat_ptr: *mut [u8], align: usize) -> StoredCPUPtr {
        StoredCPUPtr { fat_ptr, align }
    }
}

/// Used to store pointers.
///
/// Note / Safety
///
/// If the 'safe' feature isn't used, all pointers will get invalid when the drop code for an InternCPU runs as that deallocates the memory previously pointed at by the pointers stored in 'ptrs'.
#[derive(Debug, Default)]
pub struct InternCPU {
    pub ptrs: Vec<StoredCPUPtr>,
}

impl Drop for InternCPU {
    fn drop(&mut self) {
        unsafe {
            let count = get_device_count();
            *count -= 1;
            deallocate_cache(*count);
        }
    }
}

use crate::{
    deallocate_cache, get_device_count, libs::cpu::CPUCache, AsDev, BaseDevice,
    Buffer, CDatatype, CacheBuf, ClearBuf, Alloc, GenericBlas, ManualMem, VecRead, WriteBuf, Device, DeviceType
};
use std::{ffi::c_void, fmt::Debug, alloc::{Layout, handle_alloc_error}, mem::size_of};

#[derive(Debug, Default)]
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
pub struct CPU;

impl CPU {
    #[must_use]
    /// Creates an [CPU] with an InternCPU that holds an empty vector of pointers.
    pub fn new() -> CPU {
        unsafe {
            *get_device_count() += 1;
        }
        CPU
    }
}

impl Clone for CPU {
    fn clone(&self) -> Self {
        unsafe {
            *get_device_count() += 1;
        }
        Self {  }
    }
}

impl Drop for CPU {
    fn drop(&mut self) {
        unsafe {
            let count = get_device_count();
            *count -= 1;
            deallocate_cache(*count);
        }
    }
}


impl<T: Clone + Default> Alloc<T> for CPU {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        assert!(len > 0, "invalid buffer len: 0");
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = unsafe {
            std::alloc::alloc(layout)
        };

        // initialize block of memory
        for element in unsafe {std::slice::from_raw_parts_mut(ptr, len*size_of::<T>())} {
            *element = 0;
        }

        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        (ptr as *mut T, std::ptr::null_mut(), 0)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64) {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let (ptr, _, _) = self.alloc(data.len());
        let slice = unsafe {std::slice::from_raw_parts_mut(ptr, data.len())};
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
            device: self as *const CPU as *mut u8
        }
    }
}

impl AsDev for CPU {}

impl<T> ManualMem<T> for CPU {
    fn drop_buf(&self, buf: Buffer<T>) {
        unsafe {
            drop(Box::from_raw(buf.ptr.0));
        }
    }
}

impl<'a, T: Copy + Default> CacheBuf<'a, T> for CPU {
    fn cached(&'a self, len: usize) -> Buffer<'a, T> {
        CPUCache::get::<T>(self, len)
    }
}

pub fn cpu_cached<'a, T: Copy+Default>(device: &'a CPU, len: usize) -> Buffer<'a, T> {
    device.cached(len)
}

impl<T: Copy + Default> VecRead<T> for CPU {
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

impl<T: CDatatype + GenericBlas> BaseDevice<T> for CPU {}
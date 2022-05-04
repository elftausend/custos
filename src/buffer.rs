use std::ffi::c_void;

#[cfg(feature="safe")]
use crate::opencl::api::{release_mem_object, clRetainMemObject};
use crate::{Device, number::Number};


#[cfg(feature="safe")]
#[derive(Debug, Clone, Copy)]
pub enum DeallocType {
    CPU,
    CL,
    Item,
}

#[cfg(feature="safe")]
#[derive(Debug)]
pub struct Buffer<T> {
    pub ptr: *mut T,
    pub len: usize,
    pub dealloc_type: DeallocType,
}

#[cfg(feature="safe")]
unsafe impl<T> Send for Buffer<T> {}
#[cfg(feature="safe")]
unsafe impl<T> Sync for Buffer<T> {}


#[cfg(feature="safe")]
impl<T> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        if let DeallocType::CL = self.dealloc_type { 
            unsafe {
                clRetainMemObject(self.ptr as *mut c_void);
            }
        };
        Self { ptr: self.ptr, len: self.len, dealloc_type: self.dealloc_type }
    }
}

#[cfg(feature="safe")]
impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        match self.dealloc_type {
            DeallocType::CPU => unsafe {
                Box::from_raw(self.ptr);
            },
            DeallocType::CL => unsafe {
                release_mem_object(self.ptr as *mut c_void).unwrap()
            },
            _ => {}
        }
    }
}

#[cfg(not(feature="safe"))]
#[derive(Debug, Clone, Copy)]
pub struct Buffer<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T> Buffer<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }
    }
}

impl<T: Default+Copy> Buffer<T> {
    pub fn new<D: Device<T>>(device: &D, len: usize) -> Buffer<T> {
        Buffer {
            ptr: device.alloc(len),
            len,
            #[cfg(feature="safe")]
            dealloc_type: device.dealloc_type()
        }
    }

    pub fn item(&self) -> T {
        if self.len == 0 {
            return unsafe { *self.ptr };
        }
        T::default()
    }
}

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self { ptr: std::ptr::null_mut(), len: Default::default(), #[cfg(feature="safe")]dealloc_type: DeallocType::Item }
    }
}

impl<T: Number> From<T> for Buffer<T> {
    fn from(val: T) -> Self {
        Buffer { 
            ptr: Box::into_raw(Box::new(val)), 
            len: 0, 
            #[cfg(feature="safe")]
            dealloc_type: DeallocType::Item }
    }
}

impl<T: Clone, const N: usize> From<(&Box<dyn Device<T>>, &[T; N])> for Buffer<T> {
    fn from(device_slice: (&Box<dyn Device<T>>, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len(),
            #[cfg(feature="safe")]
            dealloc_type: device_slice.0.dealloc_type()
        }
    }
}

impl<T: Clone> From<(&Box<dyn Device<T>>, usize)> for Buffer<T> {
    fn from(device_len: (&Box<dyn Device<T>>, usize)) -> Self {
        Buffer {
            ptr: device_len.0.alloc(device_len.1),
            len: device_len.1,
            #[cfg(feature="safe")]
            dealloc_type: device_len.0.dealloc_type()
        }
    }
}


impl<T: Clone, D: Device<T>, const N: usize> From<(&D, [T; N])> for Buffer<T> {
    fn from(device_slice: (&D, [T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(&device_slice.1),
            len: device_slice.1.len(),
            #[cfg(feature="safe")]
            dealloc_type: device_slice.0.dealloc_type()
        }
        
    }
}

impl<T: Clone, D: Device<T>> From<(&D, &[T])> for Buffer<T> {
    fn from(device_slice: (&D, &[T])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len(),
            #[cfg(feature="safe")]
            dealloc_type: device_slice.0.dealloc_type()
        }
        
    }
}

impl<T: Clone, D: Device<T>> From<(&D, Vec<T>)> for Buffer<T> {
    fn from(device_slice: (&D, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len,
            #[cfg(feature="safe")]
            dealloc_type: device_slice.0.dealloc_type()
        }       
    }
}

impl<T: Clone, D: Device<T>> From<(&D, &Vec<T>)> for Buffer<T> {
    fn from(device_slice: (&D, &Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len,
            #[cfg(feature="safe")]
            dealloc_type: device_slice.0.dealloc_type()
        }       
    }
}

impl<T: Number> From<(*mut T, usize)> for Buffer<T> {
    fn from(info: (*mut T, usize)) -> Self {
        Buffer {
            ptr: info.0,
            len: info.1,
            #[cfg(feature="safe")]
            dealloc_type: DeallocType::Item
        } 
    }
}

impl<T> From<(*mut c_void, usize)> for Buffer<T> {
    fn from(info: (*mut c_void, usize)) -> Self {
        Buffer {
            ptr: info.0 as *mut T,
            len: info.1,
            #[cfg(feature="safe")]
            dealloc_type: DeallocType::Item
        } 
    }
}


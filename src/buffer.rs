use std::{ffi::c_void, ptr::null_mut};

#[cfg(feature="safe")]
use crate::opencl::api::{release_mem_object, clRetainMemObject};
use crate::{Device, number::Number, GenericOCL};

#[cfg_attr(not(feature = "safe"), derive(Debug, Clone, Copy))]
#[cfg_attr(feature = "safe", derive(Debug))]
pub struct Buffer<T> {
    pub ptr: (*mut T, *mut c_void),
    pub len: usize,
}

impl<T> Buffer<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.0, self.len)
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.0, self.len)
        }
    }
}

impl<T: Default+Copy> Buffer<T> {
    pub fn new<D: Device<T>>(device: &D, len: usize) -> Buffer<T> {
        Buffer {
            ptr: device.alloc(len),
            len,
        }
    }

    pub fn item(&self) -> T {
        if self.len == 0 {
            return unsafe { *self.ptr.0 };
        }
        T::default()
    }
}


#[cfg(feature="safe")]
unsafe impl<T> Send for Buffer<T> {}
#[cfg(feature="safe")]
unsafe impl<T> Sync for Buffer<T> {}


#[cfg(feature="safe")]
impl<T> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        if !self.ptr.1.is_null() { 
            unsafe {
                clRetainMemObject(self.ptr.1);
            }
        };
        Self { ptr: self.ptr, len: self.len}
    }
}

#[cfg(feature="safe")]
impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.0.is_null() {
                Box::from_raw(self.ptr.0);
            }
            if !self.ptr.1.is_null() {
                release_mem_object(self.ptr.1).unwrap()
            }
        }
    }
}

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self { ptr: (null_mut(), null_mut()), len: Default::default() }
    }
}

impl<T: Number> From<T> for Buffer<T> {
    fn from(val: T) -> Self {
        Buffer { 
            ptr: ( Box::into_raw(Box::new(val)), null_mut() ), 
            len: 0, 
        }
    }
}

impl<T: Clone, const N: usize> From<(&Box<dyn Device<T>>, &[T; N])> for Buffer<T> {
    fn from(device_slice: (&Box<dyn Device<T>>, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len(),
        }
    }
}

impl<T: Clone> From<(&Box<dyn Device<T>>, usize)> for Buffer<T> {
    fn from(device_len: (&Box<dyn Device<T>>, usize)) -> Self {
        Buffer {
            ptr: device_len.0.alloc(device_len.1),
            len: device_len.1,
        }
    }
}


impl<T: Clone, D: Device<T>, const N: usize> From<(&D, [T; N])> for Buffer<T> {
    fn from(device_slice: (&D, [T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(&device_slice.1),
            len: device_slice.1.len(),
        }
        
    }
}

impl<T: Clone, D: Device<T>> From<(&D, &[T])> for Buffer<T> {
    fn from(device_slice: (&D, &[T])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len(),
        }
        
    }
}

impl<T: Clone, D: Device<T>> From<(&D, Vec<T>)> for Buffer<T> {
    fn from(device_slice: (&D, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len,
        }       
    }
}

impl<T: Clone> From<(Box<dyn Device<T>>, Vec<T>)> for Buffer<T> {
    fn from(device_slice: (Box<dyn Device<T>>, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len,
        }       
    }
}

impl<T: Clone, D: Device<T>> From<(&D, &Vec<T>)> for Buffer<T> {
    fn from(device_slice: (&D, &Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len,
        }       
    }
}

impl<T: Default+Copy> From<(*mut T, usize)> for Buffer<T> {
    fn from(info: (*mut T, usize)) -> Self {
        Buffer {
            ptr: (info.0, null_mut()),
            len: info.1,
        } 
    }
}

impl<T: GenericOCL> From<(*mut c_void, usize)> for Buffer<T> {
    fn from(info: (*mut c_void, usize)) -> Self {
        Buffer {
            ptr: (null_mut(), info.0),
            len: info.1,
        } 
    }
}


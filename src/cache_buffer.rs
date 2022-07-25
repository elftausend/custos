use std::{rc::Weak, ffi::c_void, fmt::Debug};
use crate::{Buffer, BufFlag};

#[derive(Debug, PartialEq, Eq)]
pub struct Valid;

pub struct CacheBuffer<T> {
    buf: Buffer<T>,
    valid: Weak<Valid>
}

impl<T: Default+Copy+Debug> Debug for CacheBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_buf())
    }
}

impl<T> CacheBuffer<T> {
    pub fn new(ptr: (*mut T, *mut c_void, u64), len: usize, valid: Weak<Valid>) -> CacheBuffer<T> {
        CacheBuffer {
            buf: Buffer {
                ptr,
                len,
                flag: BufFlag::Cache,
            },
            valid,
        }
    }

    pub fn to_buf(self) -> Buffer<T> {
        self.valid.upgrade().expect("Cached buffer is invalid.");
        self.buf
    }

    pub fn as_buf(&self) -> &Buffer<T> {
        self.valid.upgrade().expect("Cached buffer is invalid.");
        &self.buf
    }

    pub fn as_mut_buf(&mut self) -> &mut Buffer<T> {
        self.valid.upgrade().expect("Cached buffer is invalid.");
        &mut self.buf
    }
}

impl<T> std::ops::Deref for CacheBuffer<T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Self::Target {
        self.as_buf()
    }
}

impl<T> std::ops::DerefMut for CacheBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_buf()
    }
}

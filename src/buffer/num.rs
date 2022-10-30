use core::{ffi::c_void, ptr::null_mut};

use crate::{BufFlag, Buffer, Device, Node, PtrType, CloneBuf};

pub struct Num<T> {
    pub num: T,
}

impl<T> PtrType<T, 0> for Num<T> {
    unsafe fn dealloc(&mut self, _len: usize) {}

    fn ptrs(&self) -> (*const T, *mut c_void, u64) {
        (&self.num as *const T, null_mut(), 0)
    }

    fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64) {
        (&mut self.num as *mut T, null_mut(), 0)
    }

    // TODO: create new trait -> e.g. "FromPtrs", implement for all PtrType types -> add bound to cache etc
    unsafe fn from_ptrs(_ptrs: (*mut T, *mut c_void, u64)) -> Self {
        unimplemented!()
        /*Num {
            num: *ptrs.0
        }*/
    }
}

impl Device for () {
    type Ptr<U, const N: usize> = Num<U>;
    type Cache<const N: usize> = ();
}

impl<'a, T: Clone> CloneBuf<'a, T> for () {
    #[inline]
    fn clone_buf(&'a self, buf: &Buffer<'a, T, Self>) -> Buffer<'a, T, Self> {
        Buffer {
            ptr: Num { num: buf.ptr.num.clone() },
            len: buf.len,
            device: buf.device,
            flag: buf.flag,
            node: buf.node,
        }
    }
}

impl<T: crate::number::Number> From<T> for Buffer<'_, T, ()> {
    #[inline]
    fn from(ptr: T) -> Self {
        Buffer {
            ptr: Num { num: ptr },
            len: 0,
            flag: BufFlag::None,
            device: None,
            node: Node::default(),
        }
    }
}

impl<'a, T> Buffer<'a, T, ()> {
    #[inline]
    pub fn copy(&self) -> Self where T: Copy {
        Buffer {
            ptr: Num { num: self.ptr.num },
            len: self.len,
            device: self.device,
            flag: self.flag,
            node: self.node,
        }        
    }
}

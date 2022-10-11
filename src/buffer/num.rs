use std::ffi::c_void;

use crate::{PtrType, Device, Buffer, AddGraph, devices::CacheAble, BufFlag, Node};

pub struct Num<T> {
    pub num: T,
}

impl<T> PtrType<T, 0> for Num<T> {
    unsafe fn dealloc(&mut self, _len: usize) {}

    fn ptrs(&self) -> (*const T, *mut c_void, u64) {
        unimplemented!()
    }

    fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64) {
        unimplemented!()
    }

    fn from_ptrs(_ptrs: (*mut T, *mut c_void, u64)) -> Self {
        unimplemented!()
    }
}

impl<D: Device, const N: usize> CacheAble<D, N> for () {
    fn retrieve<T>(_device: &D, _len: usize, _add_node: impl AddGraph) -> Buffer<T, D, N> {
        unimplemented!()
    }
}

impl Device for () {
    type Ptr<U, const N: usize> = Num<U>;
    type Cache<const N: usize> = ();
}

impl<T: crate::number::Number> From<T> for Buffer<'_, T, ()> {
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
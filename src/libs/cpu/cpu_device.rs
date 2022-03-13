use std::fmt::Debug;

use crate::{Device, VecRead, Buffer, BaseDevice, AsDev, matrix::Matrix, libs::{cpu::{CPUCache, Node, ops::{element_wise_op_mut}}, opencl::GenericOCL}};

#[derive(Debug, Clone, Copy)]
pub struct CPU;

impl CPU {
    pub fn drop<T>(buf: Buffer<T>) {
        unsafe {    
            drop(Box::from_raw(buf.ptr));
        }
    }
}


impl <T: GenericOCL>BaseDevice<T> for CPU {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        let mut out = CPUCache::get::<T>(Node::new(lhs.dims()));
        element_wise_op_mut(lhs.as_cpu_slice(), rhs.as_cpu_slice(), out.as_cpu_slice_mut(), |x, y| x + y);
        out
    }
}


impl AsDev for CPU {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(None)
    }
}

impl <T: Default+Copy>Device<T> for CPU {
    fn alloc(&self, len: usize) -> *mut T {
        Box::into_raw(vec![T::default(); len].into_boxed_slice()) as *mut T
    }

    fn from_data(&self, data: &[T]) -> *mut T {
        Box::into_raw(data.to_vec().into_boxed_slice()) as *mut T
    }
}

/* 
impl Device for CPU {
    fn alloc<T: Default+Copy>(&self, len: usize) -> *mut T {
        Box::into_raw(vec![T::default(); len].into_boxed_slice()) as *mut T
    }

    fn from_data<T: Clone>(&self, data: &[T]) -> *mut T {
        Box::into_raw(data.to_vec().into_boxed_slice()) as *mut T
    }
}
*/

impl <T: Copy+Default>VecRead<T> for CPU {
    fn read(&self, buf: &crate::Buffer<T>) -> Vec<T> {
        unsafe {
            std::slice::from_raw_parts(buf.ptr, buf.len).to_vec()
        }
    }
}
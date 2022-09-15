#![allow(unused)]
use std::{ffi::c_void, marker::PhantomData};


pub struct CPU {}

pub struct CUDA {}
pub struct OpenCL {}

pub struct Deviceless;

pub trait DevicelessAble: Alloc {}

impl DevicelessAble for CPU {}
impl DevicelessAble for OpenCL {}

pub trait AddBuf<T>: Sized {
    fn add(&self, lhs: &Buffer<T, Self>);
}

pub trait Alloc {
    fn alloc<T>(&self, len: usize) -> (*mut T, *mut c_void, u64);
}

impl Alloc for CPU {
    fn alloc<T>(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        todo!()
    }
}

impl Alloc for OpenCL {
    fn alloc<T>(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        todo!()
    }
}

pub struct Buffer<'a, T, D> {
    ptr: (*mut T, *mut c_void, u64),
    len: usize,
    device: Option<&'a D>,
}

impl<'a, T, D> Drop for Buffer<'a, T, D> {
    fn drop(&mut self) {
        todo!()
    }
}



impl<'a, T, D> Buffer<'a, T, D> {
    pub fn new(device: &'a D, len: usize) -> Self 
    where
        D: Alloc
    {
        let ptr = device.alloc(len);
        Buffer {
            ptr,
            len,
            device: Some(device),
        }
    }
}

impl<'a, T> Buffer<'a, T, Deviceless> {
    pub fn deviceless<'b>(device: &'b impl DevicelessAble, len: usize) -> Buffer<'a, T, Deviceless> {
        Buffer {
            ptr: device.alloc(len),
            len,
            device: Some(&Deviceless),
        }
    }
}

//#[test]
fn test_structure() {
    let device = CPU {};

    let buf = Buffer::<f32, _>::new(&device, 10);

    let buf = Buffer::<i16, _>::deviceless(&device, 10);
}   
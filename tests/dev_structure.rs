#![allow(unused)]
use std::{ffi::c_void, marker::PhantomData};

use custos::{cache::{CacheReturn, CacheType}, GraphReturn, OpenCL, CPU, cpu::RawCpuBuf, Alloc};

//pub struct CPU {}

//pub struct CUDA {}
//pub struct OpenCL {}

pub trait DevicelessAble: Alloc {}

impl DevicelessAble for CPU {}
impl DevicelessAble for OpenCL {}

pub trait PtrType {
    unsafe fn alloc<T>(alloc: impl Alloc, len: usize) -> Self;
    unsafe fn dealloc<T>(&mut self, len: usize);
}

impl PtrType for RawCpuBuf {
    unsafe fn alloc<T>(alloc: impl Alloc, len: usize) -> Self {
        alloc.alloc::<T>(len);
        todo!()
    }

    unsafe fn dealloc<T>(&mut self, len: usize) {
        todo!()
    }
}

pub trait Device {
    type P: PtrType;
}

impl Device for CPU {
    type P = RawCpuBuf;
}

pub struct Buf<'a, T, D: Device> {
    ptr: D::P,
    device: &'a D,
    p: PhantomData<T>,
}

impl<'a, T, D: Device> Buf<'a, T, D> 
where &'a D: Alloc,
{
    fn new(device: &'a D) {
        let ptr = unsafe {
            D::P::alloc::<T>(device, 10)
        };
    }
}



pub trait CPUCL: GraphReturn + CacheReturn {}

impl CPUCL for CPU {}

// only, if unified mem
#[cfg(unified_cl)]
impl CPUCL for OpenCL {}

pub trait AddBuf<T, D>: Sized {
    fn add(&self, lhs: &Buffer<T, D>);
}
impl<T, D: CPUCL> AddBuf<T, D> for CPU {
    fn add(&self, lhs: &Buffer<T, D>) {}
}

#[test]
fn test_add() {
    let device = CPU::new();

    let buf = Buffer::<f32, _>::new(&device, 10);

    device.add(&buf);

    let cl = OpenCL::new(0).unwrap();

    let buf = Buffer::<f32, _>::new(&cl, 10);

    device.add(&buf);
}

/*
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
*/

pub struct Buffer<'a, T, D = ()> {
    ptr: (*mut T, *mut c_void, u64),
    len: usize,
    device: &'a D,
}

impl<'a, T, D> Drop for Buffer<'a, T, D> {
    fn drop(&mut self) {
        todo!()
    }
}

impl<'a, T, D> Buffer<'a, T, D> {
    pub fn new(device: &'a D, len: usize) -> Self
    where
        D: Alloc,
    {
        let ptr = device.alloc(len);
        Buffer { ptr, len, device }
    }
}

impl<'a, T> Buffer<'a, T> {
    pub fn deviceless<'b>(device: &'b impl DevicelessAble, len: usize) -> Buffer<'a, T> {
        Buffer {
            ptr: device.alloc(len),
            len,
            device: &(),
        }
    }
}

//#[test]
fn test_structure() {
    let device = CPU::new();

    let buf = Buffer::<f32, _>::new(&device, 10);

    let buf = Buffer::<i16, _>::deviceless(&device, 10);
}

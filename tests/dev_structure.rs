/*
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
}*/

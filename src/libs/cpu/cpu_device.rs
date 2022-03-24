use std::{fmt::Debug, cell::RefCell, rc::Rc};

use crate::{AsDev, BaseDevice, BaseOps, Buffer, Device, Gemm, libs::{cpu::{CPUCache, ops::element_wise_op_mut}, opencl::GenericOCL}, matrix::Matrix, VecRead, number::Number, Dealloc, Threaded};

use super::{TBlas, CPU_CACHE};

#[derive(Debug, Clone)]
pub struct InternCPU {
    pub cpu: Rc<RefCell<CPU2>>
}
impl InternCPU {
    pub fn new(cpu: Rc<RefCell<CPU2>>) -> InternCPU {
        InternCPU { cpu }
    }
}

impl <T: Copy+Default>Device<T> for InternCPU {
    fn alloc(&self, len: usize) -> *mut T {
        let ptr = Box::into_raw(vec![T::default(); len].into_boxed_slice()) as *mut T;
        self.cpu.borrow_mut().ptrs.push(ptr as *mut usize);
        ptr
    }

    fn with_data(&self, data: &[T]) -> *mut T {
        let ptr = Box::into_raw(data.to_vec().into_boxed_slice()) as *mut T;
        self.cpu.borrow_mut().ptrs.push(ptr as *mut usize);
        ptr
    }
}

impl <T: Copy+Default>VecRead<T> for InternCPU {
    fn read(&self, buf: Buffer<T>) -> Vec<T> {
        unsafe {
            std::slice::from_raw_parts(buf.ptr, buf.len).to_vec()
        }
    }
}

impl <T: Number>BaseOps<T> for InternCPU {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(lhs, rhs, | x, y| x+y)
    }

    fn sub(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(lhs, rhs, | x, y| x-y)
    }

    fn mul(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(lhs, rhs, | x, y| x*y)
    }
}

impl Dealloc for InternCPU {
    fn dealloc_cache() {
        CPU_CACHE.with(|cache| {
            let contents = cache.borrow().nodes.clone();
            contents.into_iter()
                .for_each(|entry| {
                    let ptr = (entry.1).0;
                    unsafe { Box::from_raw(ptr.0) };
                    cache.borrow_mut().nodes.remove(&entry.0);
                });
        });
    }
}


impl <T: TBlas+Default+Copy>Gemm<T> for InternCPU {
    fn gemm(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        let m = lhs.dims().0;
        let k = lhs.dims().1;
        let n = rhs.dims().1;

        let mut c = CPUCache::get((m, n));
        T::gemm(m, n, k, lhs.as_cpu_slice(), rhs.as_cpu_slice(), c.as_cpu_slice_mut());
        c
    }
}

impl Drop for CPU2 {
    fn drop(&mut self) {
        InternCPU::dealloc_cache();
        for ptr in &self.ptrs {
            unsafe {    
                drop(Box::from_raw(*ptr));
            }
        }
        self.ptrs.clear();
    }
}


#[derive(Debug, Clone)]
pub struct CPU2 {
    pub ptrs: Vec<*mut usize>
}

impl CPU2 {
    pub fn new() -> InternCPU {
        InternCPU::new(Rc::new(RefCell::new(CPU2 { ptrs: Vec::new() })))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CPU;

impl CPU {
    pub fn drop<T>(buf: Buffer<T>) {
        unsafe {    
            drop(Box::from_raw(buf.ptr));
        }
    }
    pub fn mt<T: Default+Copy>(self) -> (Self, Threaded<CPU>) {
        (self, Threaded::new(self))
    }
}

impl <T: TBlas+Default+Copy>Gemm<T> for CPU {
    fn gemm(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        let m = lhs.dims().0;
        let k = lhs.dims().1;
        let n = rhs.dims().1;

        let mut c = CPUCache::get((m, n));
        T::gemm(m, n, k, lhs.as_cpu_slice(), rhs.as_cpu_slice(), c.as_cpu_slice_mut());
        c
    }
}

impl <T: GenericOCL+TBlas>BaseDevice<T> for CPU {}

pub fn ew_op<T: Copy+Default, F: Fn(T, T) -> T>(lhs: Matrix<T>, rhs: Matrix<T>, f: F) -> Matrix<T> {
    let mut out = CPUCache::get::<T>(lhs.dims());
    element_wise_op_mut(lhs.as_cpu_slice(), rhs.as_cpu_slice(), out.as_cpu_slice_mut(), f);
    out
}

impl <T: Number>BaseOps<T> for CPU {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(lhs, rhs, | x, y| x+y)
    }

    fn sub(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(lhs, rhs, | x, y| x-y)
    }

    fn mul(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(lhs, rhs, | x, y| x*y)
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

    fn with_data(&self, data: &[T]) -> *mut T {
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
    fn read(&self, buf: crate::Buffer<T>) -> Vec<T> {
        unsafe {
            std::slice::from_raw_parts(buf.ptr, buf.len).to_vec()
        }
    }
}

impl Dealloc for CPU {
    fn dealloc_cache() {
        CPU_CACHE.with(|cache| {
            let contents = cache.borrow().nodes.clone();
            contents.into_iter()
                .for_each(|entry| {
                    let ptr = (entry.1).0;
                    unsafe { Box::from_raw(ptr.0) };
                    cache.borrow_mut().nodes.remove(&entry.0);
                });
        });
    }
}
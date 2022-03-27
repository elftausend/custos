use std::{fmt::Debug, cell::RefCell, rc::Rc};

use crate::{BaseOps, Buffer, Device, Gemm, libs::{cpu::{CPUCache, ops::element_wise_op_mut}, opencl::GenericOCL}, matrix::Matrix, VecRead, number::Number, Dealloc, AsDev, BaseDevice};

use super::{TBlas, CPU_CACHE};

#[derive(Debug, Clone)]
pub struct InternCPU {
    pub cpu: Rc<RefCell<CPU>>
}
impl InternCPU {
    pub fn new(cpu: Rc<RefCell<CPU>>) -> InternCPU {
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
    fn alloc_with_vec(&self, vec: Vec<T>) -> *mut T {
        let ptr = Box::into_raw(vec.into_boxed_slice()) as *mut T;
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
        ew_op(self.clone(), lhs, rhs, | x, y| x+y)
    }

    fn sub(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(self.clone(), lhs, rhs, | x, y| x-y)
    }

    fn mul(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        ew_op(self.clone(), lhs, rhs, | x, y| x*y)
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

        let mut c = CPUCache::get(self.clone(),(m, n));
        T::gemm(m, n, k, lhs.as_cpu_slice(), rhs.as_cpu_slice(), c.as_cpu_slice_mut());
        c
    }
}

impl Drop for CPU {
    fn drop(&mut self) {
        let contents = CPU_CACHE.with(|cache| {
           cache.borrow().nodes.clone()         
        });
        
        for ptr in self.ptrs.iter() {

            unsafe {    
                drop(Box::from_raw(*ptr));
            }

            contents.iter()
                .for_each(|entry| {
                    let hm_ptr = ((entry.1).0).0;

                    if &hm_ptr == ptr {
                        CPU_CACHE.with(|cache| {
                            cache.borrow_mut().nodes.remove(entry.0);
                        });                        
                    }
                });
        }

        self.ptrs.clear();
    }
}


#[derive(Debug, Clone)]
pub struct CPU {
    pub ptrs: Vec<*mut usize>
}

impl CPU {
    #[must_use]
    pub fn new() -> InternCPU {
        InternCPU::new(Rc::new(RefCell::new(CPU { ptrs: Vec::new() })))
    }
}

impl AsDev for InternCPU {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(None, Some(Rc::downgrade(&self.cpu)))
    }
}

impl <T: GenericOCL+TBlas>BaseDevice<T> for InternCPU {}

pub fn ew_op<T: Copy+Default, F: Fn(T, T) -> T>(device: InternCPU, lhs: Matrix<T>, rhs: Matrix<T>, f: F) -> Matrix<T> {
    let mut out = CPUCache::get::<T>(device, lhs.dims());
    element_wise_op_mut(lhs.as_cpu_slice(), rhs.as_cpu_slice(), out.as_cpu_slice_mut(), f);
    out
}
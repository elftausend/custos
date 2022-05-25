use std::{fmt::Debug, cell::RefCell, rc::Rc, ffi::c_void};

use crate::{BaseOps, Buffer, Device, Gemm, libs::cpu::{CPUCache, ops::element_wise_op_mut}, matrix::Matrix, VecRead, number::Number, Dealloc, AsDev, BaseDevice, AssignOps, GenericOCL, ManualMem, remove_ptr, CacheBuf};

use super::{TBlas, CPU_CACHE, assign_to_lhs};

#[derive(Debug, Clone)]
/// All traits related to mathematical operations need to be implemented for this struct in order to use them.
/// This struct is should be only created via the [CPU] struct.
/// # Example
/// ```
/// use custos::{CPU, BaseOps, VecRead, Matrix};
/// 
/// let device = CPU::new();
/// let a = Matrix::<f32>::new(&device, (5, 5));
/// let b = Matrix::from((&device, (5, 5), vec![1.3; 5*5]));
/// 
/// let out = device.add(&a, &b);
/// 
/// assert_eq!(device.read(out.as_buf()), vec![1.3; 5*5]);
/// ```
pub struct InternCPU {
    pub cpu: Rc<RefCell<CPU>>
}
impl InternCPU {
    pub fn new(cpu: Rc<RefCell<CPU>>) -> InternCPU {
        InternCPU { cpu }
    }
}

#[cfg(not(feature="safe"))]
impl<T: Copy+Default> Device<T> for InternCPU {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void) {
        assert!(len > 0, "invalid buffer len: 0");
        let ptr = Box::into_raw(vec![T::default(); len].into_boxed_slice()) as *mut T;
        self.cpu.borrow_mut().ptrs.push(ptr as *mut usize);
        (ptr, std::ptr::null_mut())
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void) {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let ptr = Box::into_raw(data.to_vec().into_boxed_slice()) as *mut T;
        self.cpu.borrow_mut().ptrs.push(ptr as *mut usize);
        (ptr, std::ptr::null_mut())
    }
    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void) {
        assert!(!vec.is_empty(), "invalid buffer len: 0");
        let ptr = Box::into_raw(vec.into_boxed_slice()) as *mut T;
        self.cpu.borrow_mut().ptrs.push(ptr as *mut usize);
        (ptr, std::ptr::null_mut())
    }

    fn drop(&mut self, buf: Buffer<T>) {
        let ptrs = &mut self.cpu.borrow_mut().ptrs;
        remove_ptr(ptrs, buf.ptr.0 as *mut usize);
        self.drop_buf(buf)
    }
}

#[cfg(feature="safe")]
impl<T: Copy+Default> Device<T> for InternCPU {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void) {
        assert!(len > 0, "invalid buffer len: 0");
        (Box::into_raw(vec![T::default(); len].into_boxed_slice()) as *mut T, std::ptr::null_mut())
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void) {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        (Box::into_raw(data.to_vec().into_boxed_slice()) as *mut T, std::ptr::null_mut())
    }

    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void) {
        assert!(!vec.is_empty(), "invalid buffer len: 0");
        (Box::into_raw(vec.into_boxed_slice()) as *mut T, std::ptr::null_mut())
    }
    fn drop(&mut self, buf: Buffer<T>) {
        let ptrs = &mut self.cpu.borrow_mut().ptrs;
        remove_ptr(ptrs, buf.ptr.0 as *mut usize);
        self.drop_buf(buf)
    }

}

impl<T> ManualMem<T> for InternCPU {
    fn drop_buf(&self, buf: Buffer<T>) {
        unsafe {
            Box::from_raw(buf.ptr.0);
        }
    }
}

impl<T: Copy+Default> CacheBuf<T> for InternCPU {
    fn cached_buf(&self, len: usize) -> Buffer<T> {
        CPUCache::get::<T>(self.clone(), len)
    }
}

impl<T: Copy+Default> VecRead<T> for InternCPU {
    fn read(&self, buf: &Buffer<T>) -> Vec<T> {
        unsafe {
            std::slice::from_raw_parts(buf.ptr.0, buf.len).to_vec()
        }
    }
}

impl<T: Number> AssignOps<T> for InternCPU {
    fn add_assign(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>) {
        assign_op(lhs, rhs, |x, y| *x += y)
    }

    fn sub_assign(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>) {
        assign_op(lhs, rhs, |x, y| *x -= y)
    }
}

impl<T: Number> BaseOps<T> for InternCPU {
    fn add(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        ew_op(self.clone(), lhs, rhs, | x, y| x+y)
    }

    fn sub(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        ew_op(self.clone(), lhs, rhs, | x, y| x-y)
    }

    fn mul(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        ew_op(self.clone(), lhs, rhs, | x, y| x*y)
    }

    fn div(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        ew_op(self.clone(), lhs, rhs, | x, y| x/y)
    }

    fn clear(&self, matrix: &mut Matrix<T>) {
        for value in matrix.as_mut_slice() {
            *value = T::zero();
        }
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


impl<T: TBlas+Default+Copy> Gemm<T> for InternCPU {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(lhs.dims().1 == rhs.dims().0);
        let m = lhs.dims().0;
        let k = lhs.dims().1;
        let n = rhs.dims().1;

        let mut c = CPUCache::get(self.clone(), m*n);
        T::gemm(m, n, k, lhs.as_slice(), rhs.as_slice(), c.as_mut_slice());
        (c, (m, n)).into()
    }
}

#[derive(Debug, Clone)]
/// If the 'safe' feature isn't used, pointers are stored in the 'ptrs' field.
/// It is used to get an [InternCPU], which gives you access to all functions that were implemented for the InternCPU struct.
/// 
/// # Note / Safety
/// 
/// If the 'safe' feature isn't used, all pointers will get invalid when the drop code for a CPU object is run as that deallocates the memory previously pointed at by the pointers stored in 'ptrs'.
/// 
/// # Example
/// ```
/// use custos::{CPU, BaseOps, VecRead, Matrix};
/// 
/// let device = CPU::new();
/// let a = Matrix::<f32>::new(&device, (5, 5));
/// let b = Matrix::from((&device, (5, 5), vec![1.3; 5*5]));
/// 
/// let out = device.add(&a, &b);
/// 
/// assert_eq!(device.read(out.as_buf()), vec![1.3; 5*5]);
/// ```
pub struct CPU {
    pub ptrs: Vec<*mut usize>
}

impl CPU {
    #[must_use]
    /// Creates an [InternCPU] instance with an CPU that holds an empty vector of pointers.
    pub fn new() -> InternCPU {
        InternCPU::new(Rc::new(RefCell::new(CPU { ptrs: Vec::new() })))
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



impl AsDev for InternCPU {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(None, Some(Rc::downgrade(&self.cpu)))
    }
}

impl<T: GenericOCL+TBlas> BaseDevice<T> for InternCPU {}

pub fn assign_op<T: Copy+Default, F: Fn(&mut T, T)>(lhs: &mut Matrix<T>, rhs: &Matrix<T>, f: F) {
    assign_to_lhs(lhs.as_mut_slice(), rhs.as_slice(), f)
}

pub fn ew_op<T: Copy+Default, F: Fn(T, T) -> T>(device: InternCPU, lhs: &Matrix<T>, rhs: &Matrix<T>, f: F) -> Matrix<T> {
    let mut out = CPUCache::get::<T>(device, lhs.size());
    element_wise_op_mut(lhs.as_slice(), rhs.as_slice(), out.as_mut_slice(), f);
    (out, lhs.dims()).into()
}

pub fn each_op<T: Copy+Default, F: Fn(T) -> T>(device: &InternCPU, x: &Matrix<T>, f: F) -> Matrix<T> {
    let mut y = CPUCache::get::<T>(device.clone(), x.size());
    let slice = x.as_slice();
    for (idx, value) in y.as_mut_slice().iter_mut().enumerate() {
        *value = f(slice[idx]);
    }
    (y, x.dims()).into()
}

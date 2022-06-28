use std::{fmt::Debug, cell::RefCell, rc::Rc, ffi::c_void};
use crate::{BaseOps, Buffer, Device, Gemm, libs::cpu::{CPUCache, ops::element_wise_op_mut}, matrix::Matrix, VecRead, number::Number, AsDev, BaseDevice, AssignOps, CDatatype, ManualMem, CacheBuf, GenericBlas};
use super::{CPU_CACHE, assign_to_lhs};

#[derive(Debug, Clone, Default)]
/// A CPU is used to perform calculations on the host CPU.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CPU].
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
/// assert_eq!(device.read(&out), vec![1.3; 5*5]);
/// ```
pub struct CPU {
    pub inner: Rc<RefCell<InternCPU>>,
}

impl CPU {
    #[must_use]
    /// Creates an [CPU] with an InternCPU that holds an empty vector of pointers.
    pub fn new() -> CPU {
        CPU { 
            inner: Rc::new(RefCell::new(InternCPU {ptrs: Vec::new()})),
        }
    }
}

impl From<Rc<RefCell<InternCPU>>> for CPU {
    fn from(inner: Rc<RefCell<InternCPU>>) -> Self {
        CPU { inner }
    }
}

#[cfg(not(feature="safe"))]
impl<T: Copy+Default> Device<T> for CPU {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        assert!(len > 0, "invalid buffer len: 0");
        let ptr = Box::into_raw(vec![T::default(); len].into_boxed_slice());
        //let size = std::mem::size_of::<T>() * len;
        self.inner.borrow_mut().ptrs.push(StoredCPUPtr::new(ptr as *mut [u8], std::mem::size_of::<T>()));
        (ptr as *mut T, std::ptr::null_mut(), 0)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64) {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let ptr = Box::into_raw(data.to_vec().into_boxed_slice());
        self.inner.borrow_mut().ptrs.push(StoredCPUPtr::new(ptr as *mut [u8], std::mem::size_of::<T>()));
        (ptr as *mut T, std::ptr::null_mut(), 0)
    }
    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void, u64) {
        assert!(!vec.is_empty(), "invalid buffer len: 0");
        let ptr = Box::into_raw(vec.into_boxed_slice());
        self.inner.borrow_mut().ptrs.push(StoredCPUPtr::new(ptr as *mut [u8], std::mem::size_of::<T>()));
        (ptr as *mut T, std::ptr::null_mut(), 0)
    }

    fn drop(&mut self, buf: Buffer<T>) {
        let ptrs = &mut self.inner.borrow_mut().ptrs;
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.ptr.0, buf.len) };
        
        crate::remove_value(
        ptrs, 
    &StoredCPUPtr::new(
            slice as *mut [T] as *mut [u8], 
            std::mem::size_of::<T>()
                )
        ).unwrap();
        self.drop_buf(buf)
    }
}

#[cfg(feature="safe")]
impl<T: Clone+Default> Device<T> for CPU {
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64) {
        assert!(len > 0, "invalid buffer len: 0");
        let ptr = Box::into_raw(vec![T::default(); len].into_boxed_slice()) as *mut T;
        (ptr, std::ptr::null_mut(), 0)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64) {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        let ptr = Box::into_raw(data.to_vec().into_boxed_slice()) as *mut T;
        (ptr, std::ptr::null_mut(), 0)
    }
    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void, u64) {
        assert!(!vec.is_empty(), "invalid buffer len: 0");
        let ptr = Box::into_raw(vec.into_boxed_slice()) as *mut T;
        (ptr, std::ptr::null_mut(), 0)
    }
}

impl AsDev for CPU {
    fn as_dev(&self) -> crate::Dev {
        crate::Dev::new(None, Some(Rc::downgrade(&self.inner)), None)
    }
}

impl<T> ManualMem<T> for CPU {
    fn drop_buf(&self, buf: Buffer<T>) {
        unsafe {
            Box::from_raw(buf.ptr.0);
        }
    }
}

impl<T: Copy+Default> CacheBuf<T> for CPU {
    fn cached_buf(&self, len: usize) -> Buffer<T> {
        CPUCache::get::<T>(self, len)
    }
}

impl<T: Copy+Default> VecRead<T> for CPU {
    fn read(&self, buf: &Buffer<T>) -> Vec<T> {
        unsafe {
            std::slice::from_raw_parts(buf.ptr.0, buf.len).to_vec()
        }
    }
}

impl<T: Number> AssignOps<T> for CPU {
    fn add_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        assign_op(lhs, rhs, |x, y| *x += y)
    }

    fn sub_assign(&self, lhs: &mut Buffer<T>, rhs: &Buffer<T>) {
        assign_op(lhs, rhs, |x, y| *x -= y)
    }
}

impl<T: Number> BaseOps<T> for CPU {
    fn add(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        ew_op(self, lhs, rhs, | x, y| x+y)
    }

    fn sub(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        ew_op(self, lhs, rhs, | x, y| x-y)
    }

    fn mul(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        ew_op(self, lhs, rhs, | x, y| x*y)
    }

    fn div(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        ew_op(self, lhs, rhs, | x, y| x/y)
    }

    fn clear(&self, buf: &mut Buffer<T>) {
        for value in buf.as_mut_slice() {
            *value = T::zero();
        }
    }
}

impl<T: GenericBlas+Default+Copy> Gemm<T> for CPU {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(lhs.dims().1 == rhs.dims().0);
        let m = lhs.dims().0;
        let k = lhs.dims().1;
        let n = rhs.dims().1;

        let mut c = CPUCache::get(self, m*n);
        T::gemm(m, n, k, lhs, rhs, &mut c);
        (c, (m, n)).into()
    }
}


impl<T: CDatatype+GenericBlas> BaseDevice<T> for CPU {}

pub fn assign_op<T: Copy+Default, F: Fn(&mut T, T)>(lhs: &mut Buffer<T>, rhs: &Buffer<T>, f: F) {
    assign_to_lhs(lhs, rhs, f)
}

pub fn ew_op<T: Copy+Default, F: Fn(T, T) -> T>(device: &CPU, lhs: &Matrix<T>, rhs: &Matrix<T>, f: F) -> Matrix<T> {
    let mut out = CPUCache::get::<T>(device, lhs.size());
    element_wise_op_mut(lhs, rhs, &mut out, f);
    (out, lhs.dims()).into()
}

pub fn each_op<T: Copy+Default, F: Fn(T) -> T>(device: &CPU, x: &Matrix<T>, f: F) -> Matrix<T> {
    let mut y = CPUCache::get::<T>(device, x.size());
    
    for (idx, value) in y.iter_mut().enumerate() {
        *value = f(x[idx]);
    }
    (y, x.dims()).into()
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StoredCPUPtr {
    fat_ptr: *mut [u8],
    type_size: usize,
}

impl StoredCPUPtr {
    pub fn new(fat_ptr: *mut [u8], type_size: usize) -> StoredCPUPtr {
        StoredCPUPtr {
            fat_ptr, type_size
        }
    }
}

/// Used to store pointers.
/// 
/// Note / Safety
/// 
/// If the 'safe' feature isn't used, all pointers will get invalid when the drop code for an InternCPU runs as that deallocates the memory previously pointed at by the pointers stored in 'ptrs'.
#[derive(Debug, Default)]
pub struct InternCPU {
    pub ptrs: Vec<StoredCPUPtr>,
}


impl Drop for InternCPU {
    fn drop(&mut self) {
        let contents = CPU_CACHE.with(|cache| {
           cache.borrow().nodes.clone()         
        });
        
        for ptr in self.ptrs.iter() {
            
            unsafe {
                let len = (&*ptr.fat_ptr).len();
                let slice = std::slice::from_raw_parts_mut(ptr.fat_ptr as *mut u8, len*ptr.type_size);
                drop(Box::from_raw(slice));
                //println!("u8 slice: {slice:?}");
                //let layout = Layout::new::<u8>();
                //dealloc(slice as *mut [u8] as *mut u8, layout);
                //let slice = slice.align_to_mut::<i32>();
                //println!("slice.1: {:?}", slice.1);                
                //drop(Box::from_raw(ptr.fat_ptr as *mut usize));
            }
            
            for entry in &contents {
                let hm_ptr = ((entry.1).0).0;
                if hm_ptr == ptr.fat_ptr as *mut usize {
                    CPU_CACHE.with(|cache| {
                        cache.borrow_mut().nodes.remove(entry.0);
                    });
                }
            }
        }
        self.ptrs.clear();
    }
}
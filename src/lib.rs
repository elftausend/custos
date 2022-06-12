use std::{cell::RefCell, rc::Weak, ffi::c_void};

//pub use libs::*;
pub use buffer::*;
pub use count::*;
pub use libs::*;

pub use libs::cuda::{CudaDevice, InternCudaDevice};
#[cfg(feature="opencl")]
pub use libs::opencl::{CLDevice, InternCLDevice};
pub use libs::cpu::{CPU, InternCPU};
pub use matrix::*;

pub mod libs;

mod count;
mod buffer;

pub mod number;
mod matrix;

pub struct Error {
    pub error: Box<dyn std::error::Error>,
}

impl <E: std::error::Error + PartialEq + 'static>PartialEq<E> for Error {
    fn eq(&self, other: &E) -> bool {
        let e = self.error.downcast_ref::<E>();
        if let Some(e) = e {
            return e == other;
        }
        false
    }
}

impl From<Error> for Box<dyn std::error::Error> {
    fn from(e: Error) -> Self {
        e.error
    }
}

impl Error {
    pub fn kind<E: std::error::Error + PartialEq + 'static>(&self) -> Option<&E> {
        self.error.downcast_ref::<E>()
    }
}

impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.error)?;
        Ok(())
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)?;
        Ok(())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// ```
pub trait Device<T> {
    /// Allocate memory
    /// # Example
    /// ```
    /// use custos::{CPU, Device, Buffer, VecRead};
    /// 
    /// let device = CPU::new();
    /// let ptrs: (*mut f32, *mut std::ffi::c_void, u64) = device.alloc(12);
    /// 
    /// let buf = Buffer {
    ///     ptr: ptrs,
    ///     len: 12
    /// };
    /// assert_eq!(vec![0.; 12], device.read(&buf));
    /// ```
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64);

    /// Allocate new memory with data
    /// # Example
    /// ```
    /// use custos::{CPU, Device, Buffer, VecRead};
    /// 
    /// let device = CPU::new();
    /// let ptrs: (*mut u8, *mut std::ffi::c_void, u64) = device.with_data(&[1, 5, 4, 3, 6, 9, 0, 4]);
    /// 
    /// let buf = Buffer {
    ///     ptr: ptrs,
    ///     len: 8
    /// };
    /// assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
    /// ```
    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64);
    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void, u64) {
        self.with_data(&vec)
    }
    /// Frees the specified buffer. The pointer is removed from the pointers vector of a device.
    fn drop(&mut self, buf: Buffer<T>);
}

///All 'base' traits?
pub trait BaseDevice<T>: Device<T> + BaseOps<T> + VecRead<T> + Gemm<T> {}

/// Assignment operations
/// # Examples
/// ```
/// use custos::{CPU, Matrix, AssignOps, VecRead};
/// 
/// let device = CPU::new();
/// let mut lhs = Matrix::from((&device, 2, 2, [3, 5, 4, 1]));
/// let rhs = Matrix::from((&device, 2, 2, [1, 8, 6, 2]));
/// 
/// device.add_assign(&mut lhs, &rhs);
/// assert_eq!(vec![4, 13, 10, 3], device.read(lhs.as_buf()));
/// 
/// device.sub_assign(&mut lhs, &rhs);
/// assert_eq!(vec![3, 5, 4, 1], device.read(lhs.as_buf()));
/// ```
pub trait AssignOps<T> {
    fn add_assign(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>);
    fn sub_assign(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>);
}

#[cfg_attr(feature = "safe", doc = "```ignore")]
/// Element-wise +, -, *, / operations for matrices.
/// 
/// # Examples
/// ```
/// use custos::{CPU, Matrix, AsDev};
/// 
/// let device = CPU::new().select();
/// let a = Matrix::from((&device, (2, 3), [2, 4, 6, 8, 10, 12]));
/// let b = Matrix::from((&device, (2, 3), [12, 4, 3, 1, -5, -3]));
/// 
/// let c = a + b;
/// assert_eq!(c.read(), vec![14, 8, 9, 9, 5, 9]);
/// 
/// use custos::BaseOps;
/// let sub = device.sub(&a, &b);
/// assert_eq!(sub.read(), vec![-10, 0, 3, 7, 15, 15]);
/// ```
pub trait BaseOps<T> {
    /// Element-wise addition
    /// # Example
    /// ```
    /// use custos::{CPU, Matrix, AsDev};
    /// 
    /// let device = CPU::new().select();
    /// let a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// let b = Matrix::from((&device, 2, 3, [12, 4, 3, 1, -5, -3]));
    /// 
    /// let c = a + b;
    /// assert_eq!(c.read(), vec![14, 8, 9, 9, 5, 9]);
    /// ```
    fn add(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    
    /// Element-wise subtraction
    /// # Example
    /// ```
    /// use custos::{CPU, Matrix, AsDev, BaseOps};
    /// 
    /// let device = CPU::new().select();
    /// let a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// let b = Matrix::from((&device, 2, 3, [12, 4, 3, 1, -5, -3]));
    /// 
    /// let sub = device.sub(&a, &b);
    /// assert_eq!(sub.read(), vec![-10, 0, 3, 7, 15, 15]);
    /// ```
    fn sub(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;

    /// Element-wise multiplication
    /// # Example
    /// ```
    /// use custos::{CPU, Matrix, AsDev, BaseOps};
    /// 
    /// let device = CPU::new().select();
    /// let a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// let b = Matrix::from((&device, 2, 3, [12, 4, 3, 1, -5, -3]));
    /// 
    /// let mul = a * b;
    /// assert_eq!(mul.read(), vec![24, 16, 18, 8, -50, -36]);
    /// ```
    fn mul(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;

    /// Element-wise division
    /// # Example
    /// ```
    /// use custos::{CPU, Matrix, AsDev, BaseOps};
    /// 
    /// let device = CPU::new().select();
    /// let a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// let b = Matrix::from((&device, 2, 3, [12, 4, 3, 1, -5, -3]));
    /// 
    /// let div = device.div(&a, &b);
    /// assert_eq!(div.read(), vec![0, 1, 2, 8, -2, -4]);
    /// ```
    fn div(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;

    /// Sets all elements of the matrix to zero.
    /// # Example
    /// ```
    /// use custos::{CPU, Matrix, AsDev, BaseOps};
    /// 
    /// let device = CPU::new().select();
    /// let mut a = Matrix::from((&device, 2, 3, [2, 4, 6, 8, 10, 12]));
    /// assert_eq!(a.read(), vec![2, 4, 6, 8, 10, 12]);
    /// 
    /// device.clear(&mut a);
    /// assert_eq!(a.read(), vec![0; 6]);
    /// ```
    fn clear(&self, matrix: &mut Matrix<T>);
}

/// Trait for reading buffers.
pub trait VecRead<T> {
    /// Read the data of a buffer into a vector
    /// # Example
    /// ```
    /// use custos::{CPU, Buffer, VecRead};
    /// 
    /// let device = CPU::new();
    /// let a = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    /// let read = device.read(&a);
    /// assert_eq!(vec![1., 2., 3., 3., 2., 1.,], read);
    /// ```
    fn read(&self, buf: &Buffer<T>) -> Vec<T>;
}

/// Matrix multiplication. Uses provided device.
/// # Example
/// ```
/// use custos::{CPU, Matrix, Gemm, VecRead};
/// let device = CPU::new();
///
/// let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.,]));
/// let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.,]));
///
/// let c = device.gemm(&a, &b);
///
/// assert_eq!(device.read(c.as_buf()), vec![20., 14., 56., 41.,]);
/// ```
pub trait Gemm<T> {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
}

trait ManualMem<T> {
    fn drop_buf(&self, buf: Buffer<T>);
}

pub trait CacheBuf<T> {
    #[cfg_attr(feature = "safe", doc = "```ignore")]
    /// Adds a buffer to the cache. Following calls will return this buffer, if the corresponding internal count matches with the id used in the cache.
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev, VecRead, set_count, get_count, CacheBuf};
    /// 
    /// let device = CPU::new().select();
    /// assert_eq!(0, get_count());
    ///
    /// let mut buf = CacheBuf::<f32>::cached_buf(&device, 10);
    /// assert_eq!(1, get_count());
    ///
    /// for value in buf.as_mut_slice() {
    ///     *value = 1.5;
    /// }
    ///    
    /// set_count(0);
    /// let buf = CacheBuf::<f32>::cached_buf(&device, 10);
    /// assert_eq!(device.read(&buf), vec![1.5; 10]);
    /// ```
    fn cached_buf(&self, len: usize) -> Buffer<T>;
}


#[derive(Debug, Clone)]
pub struct Dev {
    pub cl_device: Option<Weak<RefCell<CLDevice>>>,
    pub cpu: Option<Weak<RefCell<CPU>>>,
    pub cuda: Option<Weak<RefCell<CudaDevice>>>,
}

impl Dev {
    pub fn new(
        cl_device: Option<Weak<RefCell<CLDevice>>>, 
        cpu: Option<Weak<RefCell<CPU>>>, 
        cuda: Option<Weak<RefCell<CudaDevice>>>
) -> Dev 
    {
        Dev { cl_device, cpu, cuda }
    }
}


thread_local! {
    pub static GLOBAL_DEVICE: RefCell<Dev> = RefCell::new(Dev { cl_device: None, cpu: None, cuda: None });

}

pub trait AsDev {
    fn as_dev(&self) -> Dev;
    /// Selects self as a global device. Therefore being able to use functions for matrices without specifying a compute device.
    /// When the device is dropped, the global device is no longer available.
    /// 
    /// # Example
    /// ```
    /// use custos::{CPU, BaseOps, VecRead, Matrix, AsDev};
    /// 
    /// let device = CPU::new().select();
    /// 
    /// let a = Matrix::from((&device, (5, 5), vec![1.5; 5*5]));
    /// let b = Matrix::from((&device, (5, 5), vec![1.3; 5*5]));
    /// 
    /// let out = a + b;
    /// 
    /// assert_eq!(out.read(), vec![2.8; 5*5]);
    /// ```
    #[must_use]
    fn select(self) -> Self where Self: AsDev+Clone {
        let dev = self.as_dev();
        GLOBAL_DEVICE.with(|d| *d.borrow_mut() = dev);        
        self
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceError {
    NoDeviceSelected
}

impl DeviceError {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceError::NoDeviceSelected => "No device selected, .select() on a device was not called before get_device! call",
        }
    }
}

impl core::fmt::Debug for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl core::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl From<DeviceError> for Error {
    fn from(error: DeviceError) -> Self {
        Error { error: Box::new(error) }
    }
}

impl std::error::Error for DeviceError {}

//#[cfg(feature="opencl")]
#[macro_export]
/// If a device is selected, it returns the device thus giving access to the functions implemented by the trait.
/// Therfore the trait needs to be implemented for the device.
/// 
/// # Errors
/// 
/// If no device is selected, a "NoDeviceSelected" error will be returned.
/// 
/// # Example
/// ```
/// use custos::{Error, CPU, get_device, Matrix, VecRead, AsDev, BaseOps};
/// 
/// fn main() -> Result<(), Error> {
///     let device = CPU::new().select();
///     let read = get_device!(VecRead, f32)?;
/// 
///     let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
///     let read = read.read(matrix.as_buf());
/// 
///     assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
///     let b = Matrix::from(( &device, (2, 3), [1., 1., 1., 1., 1., 1.]));
/// 
///     let base_ops = get_device!(BaseOps, f32)?;
///     let out = base_ops.add(&matrix, &b);
///     assert_eq!(out.read(), vec![2.51, 7.123, 8., 6.21, 9.62, 5.765]);
///     Ok(())
/// }
/// ```
macro_rules! get_device {
    
    ($t:ident, $g:ident) => {    
        {   

            use $crate::{GLOBAL_DEVICE, InternCPU, Error, DeviceError};
            let device: Result<Box<dyn $t<$g>>, Error> = GLOBAL_DEVICE.with(|device| {
                let device = device.borrow();
        
                let mut dev: Result<Box<dyn $t<$g>>, Error> = Err(Error::from(DeviceError::NoDeviceSelected)); 
                
                #[cfg(feature="opencl")]
                if let Some(cl) = &device.cl_device {
                    use $crate::InternCLDevice;
                    dev = Ok(Box::new(InternCLDevice::from(cl.upgrade()
                        .ok_or(Error::from(DeviceError::NoDeviceSelected))?)))
                };
    
                #[cfg(feature="cuda")]
                if let Some(cuda) = &device.cuda {
                    use $crate::InternCudaDevice;
                    dev = Ok(Box::new(InternCudaDevice::from(cuda.upgrade()
                        .ok_or(Error::from(DeviceError::NoDeviceSelected))?)))
                };
        
                if let Some(cpu) = &device.cpu {
                    dev = Ok(Box::new(InternCPU::new(cpu.upgrade()
                        .ok_or(Error::from(DeviceError::NoDeviceSelected))?)))
                };
                dev
            });
            device
        }
    }
}
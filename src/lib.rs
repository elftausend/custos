use std::{cell::RefCell, rc::Weak};

//pub use libs::*;
pub use buffer::*;
pub use count::*;
pub use libs::*;

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

impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.error)?;
        Ok(())
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.error)?;
        Ok(())
    }
}

pub trait Device<T> {
    fn alloc(&self, len: usize) -> *mut T;
    fn with_data(&self, data: &[T]) -> *mut T;
    fn alloc_with_vec(&self, vec: Vec<T>) -> *mut T {
        self.with_data(&vec)
    }
    #[cfg(feature="safe")]
    fn dealloc_type(&self) -> DeallocType;
}

pub trait OpBounds {
    
}

///All 'base' traits?
pub trait BaseDevice<T>: Device<T> + BaseOps<T> + VecRead<T> + Gemm<T> {}

pub trait AssignOps<T> {
    fn sub_assign(&self, lhs: &mut Matrix<T>, rhs: &Matrix<T>);
}

/// Base operations for matrices
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
    fn add(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    fn sub(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    fn mul(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    fn div(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
}

pub trait VecRead<T> {
    fn read(&self, buf: &Buffer<T>) -> Vec<T>;
}

pub trait Gemm<T> {
    fn gemm(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
}

pub trait Dealloc {
    fn dealloc_cache();
}

pub trait DropBuf<T> {
    fn drop_buf(&self, buf: &mut Buffer<T>);
}

#[derive(Debug, Clone)]
pub struct Dev {
    pub cl_device: Option<Weak<RefCell<CLDevice>>>,
    pub cpu: Option<Weak<RefCell<CPU>>>,
}   

impl Dev {
    pub fn new(cl_device: Option<Weak<RefCell<CLDevice>>>, cpu: Option<Weak<RefCell<CPU>>>) -> Dev {
        Dev { cl_device, cpu }
    }
    
}

thread_local! {
    pub static GLOBAL_DEVICE: RefCell<Dev> = RefCell::new(Dev { cl_device: None, cpu: None });
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

#[derive(Debug)]
pub enum DeviceError {
    NoDeviceSelected
}

impl DeviceError {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceError::NoDeviceSelected => "No device selected, .select() was not called before get_device! call in current scope",
        }
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

#[cfg(feature="opencl")]
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
///     let read = read.read(matrix.data());
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
            use $crate::{GLOBAL_DEVICE, InternCLDevice, InternCPU, Error, DeviceError};
            let device: Result<Box<dyn $t<$g>>, Error> = GLOBAL_DEVICE.with(|d| {
                let dev: Result<Box<dyn $t<$g>>, Error> = match &d.borrow().cl_device {
                    Some(cl) => Ok(Box::new(InternCLDevice::from(cl.clone().upgrade().ok_or(Error::from(DeviceError::NoDeviceSelected))?))),    
                    None => Ok(Box::new(InternCPU::new(d.borrow().cpu.as_ref().ok_or(Error::from(DeviceError::NoDeviceSelected))?.upgrade().ok_or(Error::from(DeviceError::NoDeviceSelected))?)))
                };
                dev
            });
            device
        }
    }
}

#[cfg(not(feature="opencl"))]
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
///     let read = read.read(matrix.data());
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
            let device: Result<Box<dyn $t<$g>>, Error> = GLOBAL_DEVICE.with(|d| {
                let dev: Result<Box<dyn $t<$g>>, Error> = match &d.borrow().cl_device {
                    Some(_) => Err(Error::from(DeviceError::NoDeviceSelected)),
                    None => Ok(Box::new(InternCPU::new(d.borrow().cpu.as_ref().ok_or(Error::from(DeviceError::NoDeviceSelected))?.upgrade().ok_or(Error::from(DeviceError::NoDeviceSelected))?)))
                };
                dev
            });
            device
        }
    }
}

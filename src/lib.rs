use std::{cell::RefCell, rc::Weak};

//pub use libs::*;
pub use buffer::*;
pub use count::*;
pub use libs::*;

pub use libs::{opencl::{CLDevice, InternCLDevice}, cpu::{CPU, InternCPU}};
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
}

///All 'base' traits?
pub trait BaseDevice<T>: Device<T> + BaseOps<T> + VecRead<T> + Gemm<T> {}

pub trait BaseOps<T> {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
    fn sub(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
    fn mul(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
}

pub trait VecRead<T> {
    fn read(&self, buf: Buffer<T>) -> Vec<T>;
}

pub trait Gemm<T> {
    fn gemm(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
}

pub trait Dealloc {
    fn dealloc_cache();
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
    ///selects self as global device
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

#[macro_export]
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

use std::{cell::RefCell, rc::Weak};

//pub use libs::*;
pub use buffer::*;
pub use count::*;
use libs::{cpu::{InternCPU, CPU}, opencl::{cl_device::InternCLDevice, CLDevice}};
pub use matrix::*;

pub mod libs;

mod count;
mod buffer;

pub mod number;
mod matrix;

pub struct Error {
    error: Box<dyn std::error::Error>,
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



///All 'base' traits?
pub trait BaseDevice<T>: Device<T> + BaseOps<T> + VecRead<T> + Gemm<T> {}

pub trait VecRead<T>: Device<T> {
    fn read(&self, buf: Buffer<T>) -> Vec<T>;
}

pub trait Gemm<T>: Device<T> {
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
    fn select(self) -> Self where Self: AsDev+Clone {
        let dev = self.as_dev();
        GLOBAL_DEVICE.with(|d| *d.borrow_mut() = dev);        
        self
    }
}
    
#[macro_export]
macro_rules! get_device {
    
    ($t:ident, $g:ident) => {    
        {     
            use crate::{GLOBAL_DEVICE, InternCLDevice, InternCPU};
            let dev: Box<dyn $t<$g>> = GLOBAL_DEVICE.with(|d| {
                let dev = d.borrow();
                let dev: Box<dyn $t<$g>> = match &dev.cl_device {
                    Some(cl) => Box::new(InternCLDevice::from(cl.clone().upgrade().expect("No device selected"))),
                    None => Box::new(InternCPU::new(dev.cpu.as_ref().expect("No device selected").upgrade().expect("No device selected"))),
                };
                dev
            });
            dev
        }
    }
}

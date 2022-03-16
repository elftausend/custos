pub mod libs;

mod count;
pub use count::*;
mod buffer;

pub mod number;
mod matrix;

//pub use libs::*;
pub use buffer::*;
pub use matrix::*;



use libs::{opencl::{CLDevice, GenericOCL}, cpu::{CPU, TBlas}};

pub struct Dev {
    pub cl_device: Option<CLDevice>,
}

pub trait AsDev {
    fn as_dev(&self) -> Dev;
    ///selects self as global device
    fn select(self) -> Self where Self: AsDev+Clone {
        let dev = self.as_dev();
        unsafe {
            GLOBAL_DEVICE = dev;
        }
        self
    }
}

pub static mut GLOBAL_DEVICE: Dev = Dev { cl_device: None };

/*
pub fn get_device() -> impl Device {
    match unsafe {GLOBAL_DEVICE.cl_device.clone()} {
        Some(cl_device) => cl_device,
        None => todo!()
    }
}
*/

/* 

pub fn get_device<T: GenericOCL>() -> Box<dyn BaseDevice<T>> {
    unsafe {
        match GLOBAL_DEVICE.cl_device {
            Some(cl_device) => Box::new(cl_device),
            None => Box::new(CPU)
        }
    }
}

*/

pub fn get_gemm<T: GenericOCL+TBlas>() -> Box<dyn Gemm<T>> {
    unsafe {
        match GLOBAL_DEVICE.cl_device {
            Some(cl_device) => Box::new(cl_device),
            None => Box::new(CPU)
        }
    }
}

#[macro_export]
macro_rules! get_device {
    
    ($t:ident, $g:ident) => {    
        {
            use crate::{GLOBAL_DEVICE, CPU};
            let a: Box<dyn $t<$g>> = unsafe {
                match GLOBAL_DEVICE.cl_device {
                    Some(cl_device) => Box::new(cl_device),
                    None => Box::new(CPU)
                }
            };
            a
        }

    }
}

///All base traits?
pub trait BaseDevice<T>: Device<T> + BaseOps<T> + VecRead<T> + Gemm<T> {}

//pub(crate) use get_device;

impl Dev {
    pub fn new(cl_device: Option<CLDevice>) -> Dev {
        Dev { cl_device }
    }
}

pub trait VecRead<T>: Device<T> {
    fn read(&self, buf: Buffer<T>) -> Vec<T>;
}

pub trait Gemm<T>: Device<T> {
    fn gemm(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
}
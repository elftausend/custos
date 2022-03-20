use std::{sync::Mutex, marker::PhantomData};

//pub use libs::*;
pub use buffer::*;
pub use count::*;
use libs::{cpu::CPU, opencl::CLDevice};
pub use matrix::*;

pub mod libs;

mod count;
mod buffer;

pub mod number;
mod matrix;

#[derive(Debug, Clone)]
pub struct Threaded<T, D: Dealloc<T>> {
    pub device: D,
    _pd: PhantomData<T>,
}

impl <T, D: Dealloc<T>>Threaded<T, D> {
    pub fn new(device: D) -> Threaded<T, D> {
        Threaded {
            device,
            _pd: PhantomData
        }
    }
    
}

impl <T, D: Dealloc<T>>Drop for Threaded<T, D> {
    fn drop(&mut self) {
        self.device.dealloc_cache();
    }
}


#[derive(Debug, Clone, Copy)]
pub struct Dev {
    pub cl_device: Option<CLDevice>,
}

impl Dev {
    pub fn new(cl_device: Option<CLDevice>) -> Dev {
        Dev { cl_device }
    }
}

lazy_static::lazy_static! {
    pub static ref GLOBAL_DEVICE: Mutex<Dev> = Mutex::new(Dev { cl_device: None });
}

pub trait AsDev {
    fn as_dev(&self) -> Dev;
    ///selects self as global device
    fn select(self) -> Self where Self: AsDev+Clone {
        let dev = self.as_dev();

        let mut guard = GLOBAL_DEVICE.lock().unwrap();
        *guard = dev;
        self
    }
}

#[macro_export]
macro_rules! get_device {
    
    ($t:ident, $g:ident) => {    
        {
            use crate::{CPU, GLOBAL_DEVICE};
            let guard = GLOBAL_DEVICE.lock().unwrap();
            let a: Box<dyn $t<$g>> = match guard.cl_device {
                Some(cl_device) => Box::new(cl_device),
                None => Box::new(CPU)
            };
            
            a
        }
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

pub trait Dealloc<T>: Device<T> {
    fn dealloc_cache(&self);
}
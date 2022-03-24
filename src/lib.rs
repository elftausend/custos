use std::sync::Mutex;
use std::cell::RefCell;

//pub use libs::*;
pub use buffer::*;
pub use count::*;
use libs::{cpu::{CPU, InternCPU}, opencl::{CLDevice, cl_device::InternCLDevice}};
pub use matrix::*;

pub mod libs;

mod count;
mod buffer;

pub mod number;
mod matrix;

#[derive(Debug, Clone)]
pub struct Threaded<D: Dealloc> {
    pub device: D,
}

impl <D: Dealloc>Threaded<D> {
    pub fn new(device: D) -> Threaded<D> {
        Threaded {
            device,
        }
    }
    
}

impl <D: Dealloc>Drop for Threaded<D> {
    fn drop(&mut self) {
        D::dealloc_cache();
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

pub trait Dealloc {
    fn dealloc_cache();
}




#[derive(Debug, Clone)]
pub struct Dev2 {
    pub cl_device: Option<InternCLDevice>,
    pub cpu: Option<InternCPU>,
}   

impl Dev2 {
    pub fn new(cl_device: Option<InternCLDevice>, cpu: Option<InternCPU>) -> Dev2 {
        Dev2 { cl_device, cpu }
    }
}

thread_local! {
    pub static GDEVICE: RefCell<Dev2> = RefCell::new(Dev2 { cl_device: None, cpu: None });
}

pub trait AsDev2 {
    fn as_dev(&self) -> Dev2;
    ///selects self as global device
    fn select(self) -> Self where Self: AsDev2+Clone {
        let dev = self.as_dev();
        GDEVICE.with(|d| *d.borrow_mut() = dev);        
        self
    }
}
    

#[macro_export]
macro_rules! get_device2 {
    
    ($t:ident, $g:ident) => {    
        {     
            let dev: Box<dyn $t<$g>> = GDEVICE.with(|d| {
                let dev = d.borrow();
                match &dev.cl_device {
                    Some(_) => Box::new(cl.clone()),
                    None => Box::new(dev.cpu.clone().unwrap()),
                }
            });
            dev
        }
    }
}

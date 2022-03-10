pub mod libs;
mod buffer;

//pub use libs::*;
pub use buffer::*;
use libs::{opencl::CLDevice, cpu::CPU};

pub struct Dev {
    cl_device: Option<CLDevice>,
}

pub trait AsDev {
    fn as_dev(&self) -> Dev;
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

pub fn get_device<T>() -> Box<dyn BaseDevice<T>> {
    unsafe {
        match GLOBAL_DEVICE.cl_device.clone() {
            Some(cl_device) => Box::new(cl_device),
            None => Box::new(CPU)
        }
    }
    
    
}

impl Dev {
    pub fn new(cl_device: Option<CLDevice>) -> Dev {
        Dev { cl_device}
    }
    pub fn get() -> () {
        todo!()
    }
}



pub trait VecRead<T>: Device {
    fn read(&self, buf: &Buffer<T>) -> Vec<T>;
}
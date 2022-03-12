pub mod libs;

mod buffer;

pub mod number;
mod matrix;

//pub use libs::*;
pub use buffer::*;
pub use matrix::*;



use libs::{opencl::{CLDevice, GenericOCL}, cpu::CPU};

pub struct Dev {
    cl_device: Option<CLDevice>,
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



pub fn get_device<T: GenericOCL>() -> Box<dyn BaseDevice<T>> {
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



pub trait VecRead<T>: Device<T> {
    fn read(&self, buf: &Buffer<T>) -> Vec<T>;
}
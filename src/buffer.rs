use crate::{GLOBAL_DEVICE, get_device, AsDev};


pub trait BaseDevice<T>: Device<T> {
    fn add(&self, lhs: Buffer<T>, rhs: Buffer<T>);
}

/* 
pub trait Device {
    fn alloc<T: Default+Copy>(&self, len: usize) -> *mut T;
    fn from_data<T: Clone>(&self, data: &[T]) -> *mut T;
    ///selects self as global device
    fn select(self) -> Self where Self: AsDev+Clone {
        let dev = self.as_dev();
        unsafe {
            GLOBAL_DEVICE = dev;
        }
        self
    }
}
*/

pub trait Device<T> {
    fn alloc(&self, len: usize) -> *mut T;
    fn from_data(&self, data: &[T]) -> *mut T;

}

#[derive(Debug, Clone, Copy)]
pub struct Buffer<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl <T: Default+Copy>Buffer<T> {
    pub fn new<D: Device<T>>(device: D, len: usize) -> Buffer<T> {
        Buffer {
            ptr: device.alloc(len),
            len,
        }
    }
}

impl <T: Clone, D: Device<T>, const N: usize>From<(&D, &[T; N])> for Buffer<T> {
    fn from(device_slice: (&D, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.from_data(device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}

impl <T: Clone, D: Device<T>,  const N: usize>From<(&D, [T; N])> for Buffer<T> {
    fn from(device_slice: (&D, [T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.from_data(&device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}

impl <T: Copy+Default>core::ops::Add for Buffer<T> {
    type Output = f32;

    fn add(self, rhs: Self) -> Self::Output {
        let device = get_device();
        device.add(self, rhs);
        0.
    }
}
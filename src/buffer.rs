use crate::{GLOBAL_DEVICE, Dev, get_device, AsDev};


pub trait BaseDevice<T> {
    fn add(&self, lhs: Buffer<T>, rhs: Buffer<T>);
}

pub trait Device {
    fn alloc<T: Default+Copy>(&self, len: usize) -> *mut T;
    fn from_data<T: Clone>(&self, data: &[T]) -> *mut T;
    fn select<T>(self) -> Self where Self: AsDev+Clone {
        let dev = self.as_dev();
        unsafe {
            GLOBAL_DEVICE = dev;
        }
        self
    }
}

#[derive(Debug)]
pub struct Buffer<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl <T: Default+Copy>Buffer<T> {
    pub fn new<D: Device>(device: D, len: usize) -> Buffer<T> {
        Buffer {
            ptr: device.alloc::<T>(len),
            len,
        }
    }
}

impl <D: Device, T: Clone, const N: usize>From<(&D, &[T; N])> for Buffer<T> {
    fn from(device_slice: (&D, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.from_data(device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}

impl <D: Device, T: Clone, const N: usize>From<(&D, [T; N])> for Buffer<T> {
    fn from(device_slice: (&D, [T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.from_data(&device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}

impl <T>core::ops::Add for Buffer<T> {
    type Output = f32;

    fn add(self, rhs: Self) -> Self::Output {
        let device = get_device::<T>();
        device.add(self, rhs);
        0.
    }
}
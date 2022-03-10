use crate::{GLOBAL_DEVICE, Dev};


pub trait BaseDevice<T> {
    fn add(&self, lhs: Buffer<T>, rhs: Buffer<T>);
    fn as_dev(&self) -> Dev;
}

pub trait Device {
    fn alloc<T: Default+Copy>(&self, len: usize) -> *mut T;
    fn from_data<T: Clone>(&self, data: &[T]) -> *mut T;
    fn select<T>(&self) where Self: BaseDevice<T> {
        let dev = self.as_dev();
        unsafe {
            GLOBAL_DEVICE = dev;
        }
    }
}

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
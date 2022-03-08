
pub trait Device {
    fn alloc<T>(&self, len: usize) -> *mut T;
}

pub struct Buffer<T> {
    ptr: *mut T,
    len: usize,
}

impl <T>Buffer<T> {
    pub fn new<D: Device>(device: D, len: usize) -> Buffer<T> {
        Buffer {
            ptr: device.alloc::<T>(len),
            len,
        }
        
    }
}
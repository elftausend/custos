use crate::Device;

#[derive(Debug, Clone, Copy)]
pub struct Buffer<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl <T: Default+Copy>Buffer<T> {
    pub fn new<D: Device<T>>(device: &D, len: usize) -> Buffer<T> {
        Buffer {
            ptr: device.alloc(len),
            len,
        }
    }

    pub fn item(&self) -> T {
        if self.len == 0 {
            return unsafe { *self.ptr };
        }
        T::default()
    }
}

impl <T: Copy>From<T> for Buffer<T> {
    fn from(val: T) -> Self {
        Buffer { ptr: Box::into_raw(Box::new(val)), len: 0 }
    }
}

impl <T: Clone, const N: usize>From<(&Box<dyn Device<T>>, &[T; N])> for Buffer<T> {
    fn from(device_slice: (&Box<dyn Device<T>>, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len()
        }
    }
}

impl <T: Clone>From<(&Box<dyn Device<T>>, usize)> for Buffer<T> {
    fn from(device_len: (&Box<dyn Device<T>>, usize)) -> Self {
        Buffer {
            ptr: device_len.0.alloc(device_len.1),
            len: device_len.1
        }
    }
}


impl <T: Clone, D: Device<T>, const N: usize>From<(&D, [T; N])> for Buffer<T> {
    fn from(device_slice: (&D, [T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(&device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}

impl <T: Clone, D: Device<T>>From<(&D, &[T])> for Buffer<T> {
    fn from(device_slice: (&D, &[T])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}

impl <T: Clone, D: Device<T>>From<(&D, Vec<T>)> for Buffer<T> {
    fn from(device_slice: (&D, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len
        }       
    }
}

impl <T: Clone, D: Device<T>>From<(&D, &Vec<T>)> for Buffer<T> {
    fn from(device_slice: (&D, &Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len
        }       
    }
}
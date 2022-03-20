use crate::matrix::Matrix;

pub trait BaseOps<T> {
    fn add(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
    fn sub(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
    fn mul(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
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
    fn with_data(&self, data: &[T]) -> *mut T;
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

impl <T: Clone, const N: usize>From<(Box<dyn Device<T>>, &[T; N])> for Buffer<T> {
    fn from(device_slice: (Box<dyn Device<T>>, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}


impl <T: Clone, D: Device<T>, const N: usize>From<(&D, &[T; N])> for Buffer<T> {
    fn from(device_slice: (&D, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}

impl <T: Clone, D: Device<T>,const N: usize>From<(&D, [T; N])> for Buffer<T> {
    fn from(device_slice: (&D, [T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(&device_slice.1),
            len: device_slice.1.len()
        }
        
    }
}

/* 
impl <T: Copy+Default>core::ops::Add for Buffer<T> {
    type Output = f32;

    fn add(self, rhs: Self) -> Self::Output {
        let device = get_device();
        device.add(self, rhs);
        0.
    }
}
*/
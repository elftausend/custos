use crate::{Buffer, get_device};


#[derive(Debug, Clone, Copy)]
pub struct Matrix<T> {
    data: Buffer<T>,
    dims: (usize, usize)
}

impl <T: Copy+Default>Matrix<T> {
    pub fn new(dims: (usize, usize)) -> Matrix<T> {
        let device = get_device::<T>();
        Matrix {
            data: Buffer { ptr: device.alloc(dims.0*dims.1), len: dims.0*dims.1 },
            dims,
        }
    }

    pub fn from_ptr(ptr: *mut T, dims: (usize, usize)) -> Matrix<T> {
        Matrix {
            data: Buffer {ptr, len: dims.0*dims.1},
            dims
        }
    }

    pub fn ptr(&self) -> *mut T {
        self.data.ptr
    }
    pub fn dims(&self) -> (usize, usize) {
        self.dims
    }
}
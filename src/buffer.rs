
pub trait Alloc {
    fn alloc<T>(&self, len: usize) -> *mut T;
}

pub struct Buffer<T> {
    ptr: *mut T,
    len: usize,
}

impl <T>Buffer<T> {
    pub fn new<A: Alloc>(alloc: A, len: usize) -> Buffer<T> {
        Buffer {
            ptr: alloc.alloc::<T>(len),
            len,
        }
        
    }
}
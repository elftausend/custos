
pub trait Alloc {
    fn alloc<T>() -> *mut T;
}

pub struct OpenCL {

}

impl Alloc for OpenCL {
    fn alloc<T>() -> *mut T {
        todo!()
    }
}

pub struct Buffer<T> {
    ptr: *mut T,
    len: usize,
}

impl <T>Buffer<T> {
    pub fn new() {

    }
}
use std::cell::RefCell;

use crate::number::Number;

#[cfg(feature="opencl")]
pub mod opencl;
pub mod cpu;

#[cfg(not(feature="opencl"))]
#[derive(Debug)]
pub struct CLDevice;

thread_local! {
    pub static COUNT: RefCell<usize> = RefCell::new(0);
}

pub fn set_count(count: usize) {
    COUNT.with(|c| *c.borrow_mut() = count);
}

pub fn get_count() -> usize {
    COUNT.with(|c| *c.borrow())
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Node {
    idx: usize,
    out_dims: (usize, usize),
}

impl Node {
    pub fn new(out_dims: (usize, usize)) -> Node {
        COUNT.with(|count| {
            let node = Node {
                idx: *count.borrow(),
                out_dims,
                
            };
            *count.borrow_mut() += 1;
            node
        })
    }
}

pub trait GenericOCL: Number {
    fn as_ocl_type_str() -> &'static str;
}

impl GenericOCL for f64 {
    fn as_ocl_type_str() -> &'static str {
        "double"
    }
}

impl GenericOCL for f32 {
    fn as_ocl_type_str() -> &'static str {
        "float"
    }
}

impl GenericOCL for i32 {
    fn as_ocl_type_str() -> &'static str {
        "int"
    }
}

impl GenericOCL for u32 {
    fn as_ocl_type_str() -> &'static str {
        "uint"
    }
}

impl GenericOCL for i8 {
    fn as_ocl_type_str() -> &'static str {
        "char"
    }
}

impl GenericOCL for u8 {
    fn as_ocl_type_str() -> &'static str {
        "uchar"
    }
}

impl GenericOCL for i16 {
    fn as_ocl_type_str() -> &'static str {
        "short"
    }
}
impl GenericOCL for u16 {
    fn as_ocl_type_str() -> &'static str {
        "ushort"
    }
}

impl GenericOCL for i64 {
    fn as_ocl_type_str() -> &'static str {
        "long"
    }
}

impl GenericOCL for u64 {
    fn as_ocl_type_str() -> &'static str {
        "ulong"
    }
}

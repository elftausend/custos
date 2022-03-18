pub use cl_cache::*;
pub use cl_device::CLDevice;
pub use cl_devices::*;
pub use kernel_options::*;
pub use ops::*;

use crate::number::Number;

pub mod api;

pub mod ops;
pub mod cl_device;
pub mod cl_devices;
mod kernel_options;
mod cl_cache;

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

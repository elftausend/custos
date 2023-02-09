mod impl_buffer;
mod iter;
mod macros;
mod static_devices;
mod to_device;

pub use macros::*;
pub use static_devices::*;

use crate::Device;

pub trait StaticDevice: Device {
    fn as_static() -> &'static Self;
}

impl StaticDevice for crate::CPU {
    #[inline]
    fn as_static() -> &'static Self {
        static_cpu()
    }
}

#[cfg(feature="stack")]
impl StaticDevice for crate::Stack {
    #[inline]
    fn as_static() -> &'static Self {
        &crate::Stack
    }
}

#[cfg(feature = "opencl")]
impl StaticDevice for crate::OpenCL {
    #[inline]
    fn as_static() -> &'static Self {
        static_opencl()
    }
}
#[cfg(feature = "cuda")]
impl StaticDevice for crate::CUDA {
    #[inline]
    fn as_static() -> &'static Self {
        static_cuda()
    }
}

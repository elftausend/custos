mod static_devices;
mod iter;
mod to_device;
mod macros;

pub use macros::*;
pub use static_devices::*;

pub trait StaticGPU {
    fn as_static() -> &'static Self;
}

#[cfg(feature = "opencl")]
impl StaticGPU for crate::OpenCL {
    #[inline]
    fn as_static() -> &'static Self {
        static_opencl()
    }
}
#[cfg(feature = "cuda")]
impl StaticGPU for crate::CUDA {
    #[inline]
    fn as_static() -> &'static Self {
        static_cuda()
    }
}
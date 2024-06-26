mod launch_shader;
mod ops;
mod spirv;
mod wgsl_device;

pub use launch_shader::*;
pub use spirv::*;

pub trait WgslDevice: Sized {
    fn new(idx: usize) -> crate::Result<Self>;
}

pub fn chosen_wgsl_idx() -> usize {
    std::env::var("CUSTOS_WGSL_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect(
            "Environment variable 'CUSTOS_WGSL_DEVICE_IDX' contains an invalid CUDA device index!",
        )
}

pub trait WgslNumber: crate::Number {}
impl WgslNumber for f32 {}
impl WgslNumber for i32 {}
impl WgslNumber for u32 {}
#[cfg(feature = "half")]
impl WgslNumber for half::f16 {}

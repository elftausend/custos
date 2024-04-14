mod launch_shader;
mod spirv;
mod wgsl_device;

pub use launch_shader::*;
pub use spirv::*;

pub fn chosen_wgsl_idx() -> usize {
    std::env::var("CUSTOS_WGSL_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect(
            "Environment variable 'CUSTOS_WGSL_DEVICE_IDX' contains an invalid CUDA device index!",
        )
}

use ash::{Device, vk::{ShaderModule, self}};

pub fn create_shader_module(code: &[u8], device: &Device) -> ShaderModule {
    unsafe {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code.as_ptr() as _,
            ..Default::default()
        };
        device
            .create_shader_module(&shader_module_create_info, None)
            .unwrap()
    }
}

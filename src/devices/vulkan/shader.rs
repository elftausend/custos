use ash::{
    vk::{self, DescriptorPool, DescriptorSet, Pipeline, PipelineLayout, ShaderModule, DescriptorSetLayout},
    Device,
};

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

// add dyn AsDescriptorType ..
pub fn create_descriptor_set_layout_from_desc_types(device: &Device, descriptor_types: &[vk::DescriptorType]) -> DescriptorSetLayout {
    let descriptor_set_layout_bindings = descriptor_types
        .iter()
        .copied()
        .enumerate()
        .map(
            |(binding, descriptor_type)| vk::DescriptorSetLayoutBinding {
                binding: binding as u32,
                descriptor_type,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>();

    let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
        p_bindings: descriptor_set_layout_bindings.as_ptr(),
        ..Default::default()
    };

    unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }
        .unwrap() 
}


pub struct Operation {
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_pool: DescriptorPool,
    descriptor_set: DescriptorSet,
}

pub fn cached_operation() {}

pub fn launch_shader(device: &Device, shader: &ShaderModule) {}

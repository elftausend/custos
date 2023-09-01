use core::ffi::CStr;

use ash::{
    prelude::VkResult,
    vk::{
        self, DescriptorPool, DescriptorSet, DescriptorSetLayout, Pipeline, PipelineCache,
        PipelineLayout, ShaderModule,
    },
    Device,
};

use crate::wgsl::Spirv;

pub fn create_shader_module(device: &Device, code: &[u8]) -> VkResult<ShaderModule> {
    unsafe {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code.as_ptr() as _,
            ..Default::default()
        };
        device.create_shader_module(&shader_module_create_info, None)
    }
}

// add dyn AsDescriptorType ..
pub fn create_descriptor_set_layout_from_desc_types(
    device: &Device,
    descriptor_types: &[vk::DescriptorType],
) -> VkResult<DescriptorSetLayout> {
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
}

pub fn create_pipeline(
    device: &Device,
    descriptor_set_layout: DescriptorSetLayout,
    shader_module: ShaderModule,
) -> VkResult<(Pipeline, PipelineLayout)> {
    let pipeline_layout = {
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(core::slice::from_ref(&descriptor_set_layout));
        unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }.unwrap()
    };

    // create the pipeline
    let pipeline_create_info = vk::ComputePipelineCreateInfo {
        stage: vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") }.as_ptr(),
            ..Default::default()
        },
        layout: pipeline_layout,
        ..Default::default()
    };

    let pipeline = unsafe {
        // use pipeline cache from context??
        device.create_compute_pipelines(PipelineCache::null(), &[pipeline_create_info], None)
    }
    .map_err(|(_, err)| err)?[0];
    Ok((pipeline, pipeline_layout))
}

pub fn create_descriptor_pool(device: &Device, descriptor_count: u32) -> VkResult<DescriptorPool> {
    let descriptor_pool_sizes = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count,
    };
    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
        .max_sets(1)
        .pool_sizes(core::slice::from_ref(&descriptor_pool_sizes));

    unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }
}

pub struct Operation {
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_pool: DescriptorPool,
    descriptor_set: DescriptorSet,
}

impl Operation {
    pub fn new(device: &Device, wgsl: impl AsRef<str>, descriptor_types: &[vk::DescriptorType]) {
        let spirv = Spirv::from_wgsl(wgsl).unwrap();
        let shader_module = create_shader_module(device, spirv.as_byte_slice()).unwrap();
        let descriptor_set_layout =
            create_descriptor_set_layout_from_desc_types(device, descriptor_types).unwrap();
        let (pipeline, pipeline_layout) =
            create_pipeline(device, descriptor_set_layout, shader_module).unwrap();
    }
}

pub fn cached_operation() {}

pub fn launch_shader(device: &Device, shader: &ShaderModule) {}

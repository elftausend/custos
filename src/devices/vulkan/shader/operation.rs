use ash::{vk::{Pipeline, ShaderModule, PipelineLayout, DescriptorPool, DescriptorSet, DescriptorType}, Device};

use crate::wgsl::Spirv;

use super::{create_shader_module, descriptor::{create_descriptor_set_layout_from_desc_types, create_descriptor_pool, allocate_descriptor_set}, create_pipeline};

#[derive(Debug, Clone, Copy)]
pub struct Operation {
    pub pipeline: Pipeline,
    pub shader_module: ShaderModule,
    pub pipeline_layout: PipelineLayout,
    pub descriptor_pool: DescriptorPool,
    pub descriptor_set: DescriptorSet,
}

impl Operation {
    pub fn new(
        device: &Device,
        wgsl: impl AsRef<str>,
        descriptor_types: &[DescriptorType],
    ) -> Self {
        let spirv = Spirv::from_wgsl(wgsl).unwrap();
        let shader_module = create_shader_module(device, spirv.as_byte_slice()).unwrap();
        let descriptor_set_layout =
            create_descriptor_set_layout_from_desc_types(device, descriptor_types).unwrap();
        let (pipeline, pipeline_layout) =
            create_pipeline(device, descriptor_set_layout, shader_module).unwrap();

        let descriptor_pool =
            create_descriptor_pool(device, descriptor_types.len() as u32).unwrap();

        let descriptor_set =
            allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout).unwrap();

        Operation {
            pipeline,
            shader_module,
            pipeline_layout,
            descriptor_pool,
            descriptor_set,
        }
    }
}

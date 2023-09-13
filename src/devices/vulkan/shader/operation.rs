use ash::{
    vk::{DescriptorPool, DescriptorSet, DescriptorType, Pipeline, PipelineLayout, ShaderModule},
    Device,
};

use crate::wgsl::Spirv;

use super::{
    create_pipeline, create_shader_module,
    descriptor::{
        allocate_descriptor_set, create_descriptor_pool,
        create_descriptor_set_layout_from_desc_types,
    },
};

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
    ) -> crate::Result<Self> {
        let spirv = Spirv::from_wgsl(wgsl)?;
        let shader_module = create_shader_module(device, spirv.as_slice())?;
        let descriptor_set_layout =
            create_descriptor_set_layout_from_desc_types(device, descriptor_types)?;
        let (pipeline, pipeline_layout) =
            create_pipeline(device, descriptor_set_layout, shader_module)?;

        let descriptor_pool = create_descriptor_pool(device, descriptor_types.len() as u32)?;

        let descriptor_set =
            allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;

        Ok(Operation {
            pipeline,
            shader_module,
            pipeline_layout,
            descriptor_pool,
            descriptor_set,
        })
    }
}

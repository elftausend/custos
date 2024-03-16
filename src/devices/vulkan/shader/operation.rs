use super::{
    create_shader_module,
    descriptor::{
        allocate_descriptor_set, create_descriptor_pool,
        create_descriptor_set_layout_from_desc_types,
    },
    pipeline::create_pipeline,
};
use crate::{vulkan::Context, wgsl::Spirv};
use ash::vk::{
    DescriptorPool, DescriptorSet, DescriptorSetLayout, DescriptorType, Pipeline, PipelineLayout,
    ShaderModule,
};
use std::rc::Rc;

#[derive(Clone)]
pub struct Operation {
    pub context: Rc<Context>,
    pub pipeline: Pipeline,
    pub shader_module: ShaderModule,
    pub pipeline_layout: PipelineLayout,
    pub descriptor_pool: DescriptorPool,
    pub descriptor_set: DescriptorSet,
    pub descriptor_set_layout: DescriptorSetLayout,
}

impl Operation {
    pub fn new(
        context: Rc<Context>,
        wgsl: impl AsRef<str>,
        descriptor_types: &[DescriptorType],
    ) -> crate::Result<Self> {
        let spirv = Spirv::from_wgsl(wgsl)?;
        let device = &context.device;
        let shader_module = create_shader_module(device, spirv.as_slice())?;
        let descriptor_set_layout =
            create_descriptor_set_layout_from_desc_types(device, descriptor_types)?;
        let (pipeline, pipeline_layout) =
            create_pipeline(device, descriptor_set_layout, shader_module)?;

        let descriptor_pool = create_descriptor_pool(device, descriptor_types.len() as u32)?;

        let descriptor_set =
            allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;

        Ok(Operation {
            context,
            pipeline,
            shader_module,
            pipeline_layout,
            descriptor_pool,
            descriptor_set,
            descriptor_set_layout,
        })
    }
}

impl Drop for Operation {
    #[inline]
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

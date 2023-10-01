use ash::{
    prelude::VkResult,
    vk::{self, DescriptorPool, DescriptorSet, DescriptorSetLayout},
    Device,
};

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

    let descriptor_set_layout_create_info =
        vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_set_layout_bindings);

    unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }
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

pub fn allocate_descriptor_set(
    device: &Device,
    descriptor_pool: DescriptorPool,
    descriptor_set_layout: DescriptorSetLayout,
) -> VkResult<DescriptorSet> {
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(core::slice::from_ref(&descriptor_set_layout));

    Ok(unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }?[0])
}

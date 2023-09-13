use ash::{
    prelude::VkResult,
    vk::{self, CommandBuffer, CommandPool},
    Device,
};

pub fn create_command_pool(device: &Device, compute_family_idx: usize) -> VkResult<CommandPool> {
    let command_pool_create_info = vk::CommandPoolCreateInfo {
        queue_family_index: compute_family_idx as u32,
        ..Default::default()
    };

    unsafe { device.create_command_pool(&command_pool_create_info, None) }
}

pub fn create_command_buffer(
    device: &Device,
    command_pool: CommandPool,
) -> VkResult<CommandBuffer> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    Ok(unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }?[0])
}

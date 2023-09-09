mod operation;
mod descriptor;
mod command;

use operation::Operation;
pub use descriptor::*;
pub use command::*;

use core::ffi::CStr;
use std::collections::HashMap;

use ash::{
    prelude::VkResult,
    vk::{
        self, Buffer, DescriptorSetLayout, DescriptorType, Pipeline,
        PipelineCache, PipelineLayout, ShaderModule, CommandBuffer, Fence,
    },
    Device,
};

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

// combine with other Caches
pub struct ShaderCache {
    // use hash directly (prevent &str->String?) => Use NoHasher
    cache: HashMap<String, Operation>,
}

impl ShaderCache {
    pub fn add(
        &mut self,
        device: &Device,
        src: impl AsRef<str>,
        args: &[DescriptorType],
    ) -> Operation {
        let src = src.as_ref();
        let operation = Operation::new(device, src, args);
        self.cache.insert(src.to_string(), operation);
        operation
    }
    pub fn get(
        &mut self,
        device: &Device,
        src: impl AsRef<str>,
        args: &[DescriptorType],
    ) -> Operation {
        match self.cache.get(src.as_ref()) {
            Some(operation) => *operation,
            None => self.add(device, src, args),
        }
    }
}

pub fn luanch_shader2(
    device: &Device,
    gws: [u32; 3],
    shader_cache: &mut ShaderCache,
    command_buffer: CommandBuffer,
    compute_family_idx: usize,
    src: impl AsRef<str>,
    args: &[Buffer],
) {
    let operation = shader_cache.get(
        device,
        src,
        &args
            .iter()
            .map(|_| DescriptorType::STORAGE_BUFFER)
            .collect::<Vec<_>>(),
    );
    let descriptor_infos = args
        .iter()
        .copied()
        .map(|buffer| vk::DescriptorBufferInfo {
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        })
        .collect::<Vec<_>>();

    let write_descriptor_sets = descriptor_infos
        .into_iter()
        .enumerate()
        .map(|(idx, info)| {
            vk::WriteDescriptorSet::builder()
                .dst_set(operation.descriptor_set)
                .dst_binding(idx as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&info))
                .build()
        })
        .collect::<Vec<_>>();
    unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) }


    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }.unwrap();
    unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, operation.pipeline) };
    unsafe {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            operation.pipeline_layout,
            0,
            core::slice::from_ref(&operation.descriptor_set),
            &[],
        )
    };
    unsafe { device.cmd_dispatch(command_buffer, gws[0], gws[1], gws[2]) };
    unsafe { device.end_command_buffer(command_buffer) }.unwrap();
    let queue = unsafe { device.get_device_queue(compute_family_idx as u32, 0) };
    let submit_info =
        vk::SubmitInfo::builder().command_buffers(core::slice::from_ref(&command_buffer));

    
    unsafe { device.queue_submit(queue, core::slice::from_ref(&submit_info), Fence::null()) }
        .unwrap();
    unsafe { device.device_wait_idle() }.unwrap();
}
pub fn cached_operation() {}

pub fn launch_shader(device: &Device, shader: &ShaderModule) {}

#[cfg(test)]
mod tests {
    #[test]
    fn test_launch_shader() {

    }
}

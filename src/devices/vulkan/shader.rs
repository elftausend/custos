mod command;
mod descriptor;
mod operation;

pub use command::*;
pub use descriptor::*;
pub use operation::Operation;

use core::ffi::CStr;
use std::collections::HashMap;

use ash::{
    prelude::VkResult,
    vk::{
        self, Buffer, CommandBuffer, DescriptorBufferInfo, DescriptorSet, DescriptorSetLayout,
        DescriptorType, Fence, Pipeline, PipelineCache, PipelineLayout, ShaderModule,
        WriteDescriptorSet,
    },
    Device,
};

pub fn create_shader_module(device: &Device, code: &[u32]) -> VkResult<ShaderModule> {
    unsafe {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(code);

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
        device.create_compute_pipelines(
            PipelineCache::null(),
            core::slice::from_ref(&pipeline_create_info),
            None,
        )
    }
    .map_err(|(_, err)| err)?[0];
    Ok((pipeline, pipeline_layout))
}

// combine with other Caches
#[derive(Debug, Default, Clone)]
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
    ) -> crate::Result<Operation> {
        let src = src.as_ref();
        let operation = Operation::new(device, src, args)?;
        self.cache.insert(src.to_string(), operation);
        Ok(operation)
    }

    #[inline]
    pub fn get(
        &mut self,
        device: &Device,
        src: impl AsRef<str>,
        args: &[DescriptorType],
    ) -> crate::Result<Operation> {
        match self.cache.get(src.as_ref()) {
            Some(operation) => Ok(*operation),
            None => self.add(device, src, args),
        }
    }
}

pub fn create_descriptor_infos(bufs: &[Buffer]) -> Vec<[DescriptorBufferInfo; 1]> {
    bufs.iter()
        .copied()
        .map(|buffer| {
            [vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            }]
        })
        .collect::<Vec<_>>()
}

pub fn create_write_descriptor_sets(
    descriptor_infos: &[[DescriptorBufferInfo; 1]],
    descriptor_set: DescriptorSet,
) -> Vec<WriteDescriptorSet> {
    descriptor_infos
        .into_iter()
        .enumerate()
        .map(|(idx, info)| {
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(idx as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
                .build()
        })
        .collect::<Vec<_>>()
}

pub fn launch_shader(
    device: &Device,
    gws: [u32; 3],
    shader_cache: &mut ShaderCache,
    command_buffer: CommandBuffer,
    compute_family_idx: usize,
    src: impl AsRef<str>,
    args: &[Buffer],
) -> crate::Result<()> {
    let operation = shader_cache.get(
        device,
        src,
        &args
            .iter()
            .map(|_| DescriptorType::STORAGE_BUFFER)
            .collect::<Vec<_>>(),
    )?;

    let descriptor_infos = create_descriptor_infos(args);
    let write_descriptor_sets =
        create_write_descriptor_sets(&descriptor_infos, operation.descriptor_set);

    unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) }

    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }?;
    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            operation.pipeline,
        )
    };
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
    unsafe { device.end_command_buffer(command_buffer) }?;
    let queue = unsafe { device.get_device_queue(compute_family_idx as u32, 0) };
    let submit_info =
        vk::SubmitInfo::builder().command_buffers(core::slice::from_ref(&command_buffer));

    unsafe { device.queue_submit(queue, core::slice::from_ref(&submit_info), Fence::null()) }?;

    unsafe { device.device_wait_idle() }?;
    unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty()) }?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::vulkan::{
        context::Context,
        shader::{launch_shader, ShaderCache},
        vk_array::VkArray,
    };
    use std::rc::Rc;

    #[test]
    fn test_launch_shader() {
        let context = Rc::new(Context::new(0).unwrap());
        let mut shader_cache = ShaderCache::default();
        let src = "@group(0)
            @binding(0)
            var<storage, read_write> a: array<f32>;
            
            @group(0)
            @binding(1)
            var<storage, read_write> b: array<f32>;
    
            @group(0)
            @binding(2)
            var<storage, read_write> out: array<f32>;
            
            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return;    
                }
                
                out[global_id.x] = a[global_id.x] + b[global_id.x];
            }
        ";

        let lhs = VkArray::from_slice(
            context.clone(),
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        let rhs = VkArray::from_slice(
            context.clone(),
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        let mut out =
            VkArray::<f32>::new(context.clone(), 10, crate::flag::AllocFlag::None).unwrap();

        launch_shader(
            &context.device,
            [1, 1, 1],
            &mut shader_cache,
            context.command_buffer,
            context.compute_family_idx,
            &src,
            &[lhs.buf, rhs.buf, out.buf], // &[out.buf, out2.buf]
        )
        .unwrap();

        assert_eq!(lhs.as_slice(), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        assert_eq!(rhs.as_slice(), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);

        assert_eq!(
            out.as_slice(),
            [2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]
        );

        for _ in 0..10 {
            for x in out.as_mut_slice() {
                *x = 0.;
            }
            assert_eq!(out.as_slice(), [0.; 10]);
            launch_shader(
                &context.device,
                [1, 1, 1],
                &mut shader_cache,
                context.command_buffer,
                context.compute_family_idx,
                &src,
                &[lhs.buf, rhs.buf, out.buf],
            )
            .unwrap();

            assert_eq!(lhs.as_slice(), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
            assert_eq!(rhs.as_slice(), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);

            assert_eq!(
                out.as_slice(),
                [2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]
            );
        }
    }
}

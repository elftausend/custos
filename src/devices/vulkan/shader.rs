mod command;
mod descriptor;
mod operation;
mod pipeline;
mod shader_argument;

pub use command::*;
pub use descriptor::*;
pub use operation::Operation;
pub use pipeline::*;
pub use shader_argument::*;

use std::collections::HashMap;

use ash::{
    prelude::VkResult,
    vk::{
        self, Buffer, DescriptorBufferInfo, DescriptorSet, DescriptorType, Fence, ShaderModule,
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

pub fn create_descriptor_infos(
    bufs: &[(usize, Buffer)],
) -> Vec<(usize, [DescriptorBufferInfo; 1])> {
    bufs.iter()
        .copied()
        .map(|(idx, buffer)| {
            (
                idx,
                [vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                }],
            )
        })
        .collect::<Vec<_>>()
}

pub fn create_write_descriptor_sets(
    descriptor_infos: &[(usize, [DescriptorBufferInfo; 1])],
    descriptor_set: DescriptorSet,
) -> Vec<WriteDescriptorSet> {
    descriptor_infos
        .into_iter()
        .map(|(idx, info)| {
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(*idx as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
                .build()
        })
        .collect::<Vec<_>>()
}

pub fn launch_shader(
    context: std::rc::Rc<super::Context>,
    gws: [u32; 3],
    shader_cache: &mut ShaderCache,
    src: impl AsRef<str>,
    args: &[&dyn AsVkShaderArgument],
) -> crate::Result<()> {
    let device = &context.device;
    let command_buffer = context.command_buffer;

    let args = args
        .iter()
        .map(|arg| arg.as_arg(context.clone()))
        .collect::<Vec<_>>();

    let operation = shader_cache.get(
        device,
        src,
        &args
            .iter()
            .map(|arg| arg.descriptor_type)
            .collect::<Vec<_>>(),
    )?;

    let buffer_args = args
        .iter()
        .enumerate()
        .filter_map(|(idx, arg)| {
            if arg.descriptor_type != DescriptorType::STORAGE_BUFFER {
                None
            } else {
                Some((idx, arg.buffer))
            }
        })
        .collect::<Vec<_>>();

    let descriptor_infos = create_descriptor_infos(&buffer_args);
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
    let queue = unsafe { device.get_device_queue(context.compute_family_idx as u32, 0) };
    let submit_info =
        vk::SubmitInfo::builder().command_buffers(core::slice::from_ref(&command_buffer));

    unsafe { device.queue_submit(queue, core::slice::from_ref(&submit_info), Fence::null()) }?;

    unsafe { device.device_wait_idle() }?;
    unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty()) }?;
    drop(args);
    Ok(())
}

#[cfg(test)]
mod tests {
    use ash::vk::BufferUsageFlags;

    use crate::vulkan::{
        context::Context,
        shader::{launch_shader, ShaderCache},
        vk_array::VkArray,
    };
    use std::rc::Rc;

    #[test]
    fn test_launch_vk_shader_with_num_as_arg() {
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

            @group(0)
            @binding(3)
            var<storage, read> bias: array<f32>;
            
            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return;    
                }
                
                out[global_id.x] = a[global_id.x] + b[global_id.x] + bias[0];
            }
        ";

        let lhs = VkArray::from_slice(
            context.clone(),
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        let rhs = VkArray::from_slice(
            context.clone(),
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        let out = VkArray::<f32>::new(
            context.clone(),
            10,
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();

        launch_shader(
            context.clone(),
            [1, 1, 1],
            &mut shader_cache,
            &src,
            &[&lhs.buf, &rhs.buf, &out.buf, &1f32], // &[out.buf, out2.buf]
        )
        .unwrap();

        assert_eq!(lhs.as_slice(), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        assert_eq!(rhs.as_slice(), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);

        assert_eq!(
            out.as_slice(),
            [3., 5., 7., 9., 11., 13., 15., 17., 19., 21.]
        );
    }

    #[test]
    fn test_launch_vk_shader_multiple_times() {
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
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        let rhs = VkArray::from_slice(
            context.clone(),
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        let mut out = VkArray::<f32>::new(
            context.clone(),
            10,
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();

        launch_shader(
            context.clone(),
            [1, 1, 1],
            &mut shader_cache,
            &src,
            &[&lhs.buf, &rhs.buf, &out.buf], // &[out.buf, out2.buf]
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
                context.clone(),
                [1, 1, 1],
                &mut shader_cache,
                &src,
                &[&lhs.buf, &rhs.buf, &out.buf],
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

    #[test]
    fn test_launch_vk_shader() {
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
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        let rhs = VkArray::from_slice(
            context.clone(),
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        let out = VkArray::<f32>::new(
            context.clone(),
            10,
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();

        launch_shader(
            context.clone(),
            [1, 1, 1],
            &mut shader_cache,
            &src,
            &[&lhs.buf, &rhs.buf, &out.buf], // &[out.buf, out2.buf]
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

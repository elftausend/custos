use std::rc::Rc;

use ash::vk::{self, BufferUsageFlags, DescriptorType, Fence};
use custos::{
    vulkan::{
        create_descriptor_infos, create_write_descriptor_sets, Context, ShaderCache, VkArray,
    },
    HostPtr,
};

#[test]
fn test_with_custos_comps() {
    let context = Rc::new(Context::new(0).unwrap());

    let lhs = VkArray::from_slice(
        context.clone(),
        &[1f32, 2., 3., 4., 5., 6., 7.],
        BufferUsageFlags::STORAGE_BUFFER,
        custos::flag::AllocFlag::None,
    )
    .unwrap();
    let rhs = VkArray::from_slice(
        context.clone(),
        &[1f32, 2., 3., 4., 5., 6., 7.],
        BufferUsageFlags::STORAGE_BUFFER,
        custos::flag::AllocFlag::None,
    )
    .unwrap();

    let out = VkArray::<f32>::new(
        context.clone(),
        lhs.len,
        BufferUsageFlags::STORAGE_BUFFER,
        custos::flag::AllocFlag::None,
    )
    .unwrap();

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
            @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return;    
                }
                out[global_id.x] = a[global_id.x] + b[global_id.x];
            }
    ";
    let operation = shader_cache
        .get(
            &context.device,
            src,
            &[
                DescriptorType::STORAGE_BUFFER,
                DescriptorType::STORAGE_BUFFER,
                DescriptorType::STORAGE_BUFFER,
            ],
        )
        .unwrap();
    // let operation = Operation::new(&context.device, &src, &[DescriptorType::STORAGE_BUFFER, DescriptorType::STORAGE_BUFFER, DescriptorType::STORAGE_BUFFER]);

    let descriptor_infos = create_descriptor_infos(&[(0, lhs.buf), (1, rhs.buf), (2, out.buf)]);
    let write_descriptor_sets =
        create_write_descriptor_sets(&descriptor_infos, operation.descriptor_set);
    /*let descriptor_buffer_info = [vk::DescriptorBufferInfo {
        buffer: lhs.buf,
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let descriptor_buffer_info1 = [vk::DescriptorBufferInfo {
        buffer: rhs.buf,
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let descriptor_buffer_info2 = [vk::DescriptorBufferInfo {
        buffer: out.buf,
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let write_descriptor_set = vk::WriteDescriptorSet::builder()
        .dst_set(operation.descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&descriptor_buffer_info);
    let write_descriptor_set1 = vk::WriteDescriptorSet::builder()
        .dst_set(operation.descriptor_set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&descriptor_buffer_info1);

    let write_descriptor_set2 = vk::WriteDescriptorSet::builder()
        .dst_set(operation.descriptor_set)
        .dst_binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&descriptor_buffer_info2);
    let write_descriptor_sets = [
        write_descriptor_set.build(),
        write_descriptor_set1.build(),
        write_descriptor_set2.build(),
    ];*/

    let device = &context.device;

    unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };
    let command_buffer = context.command_buffer;

    // make a command buffer that runs the compute shader
    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }.unwrap();
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
    unsafe { device.cmd_dispatch(command_buffer, 1, 1, 1) };
    unsafe { device.end_command_buffer(command_buffer) }.unwrap();

    // run it and wait until it is completed
    let queue = unsafe { device.get_device_queue(context.compute_family_idx as u32, 0) };
    let submit_info =
        vk::SubmitInfo::builder().command_buffers(core::slice::from_ref(&command_buffer));

    unsafe { device.queue_submit(queue, core::slice::from_ref(&submit_info), Fence::null()) }
        .unwrap();
    unsafe { device.device_wait_idle() }.unwrap();

    println!("out: {:?}", out.as_slice());
}

use std::{ffi::CStr, mem::size_of_val, ptr, time::Instant};

use ash::{
    vk::{
        self, Buffer, DeviceMemory, Fence, Instance, InstanceCreateInfo, PhysicalDevice,
        PipelineCache, StructureType,
    },
    Entry,
};
use naga::back::spv::{Options, PipelineOptions};

fn get_memory_type_index(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    property_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..memory_properties.memory_type_count {
        let mt = &memory_properties.memory_types[i as usize];
        if (memory_type_bits & (1 << i)) != 0 && mt.property_flags.contains(property_flags) {
            return Some(i);
        }
    }
    None
}

#[test]
fn test_vulkan_compute_with_wgsl_and_spirv() {
    let entry = unsafe { Entry::load().unwrap() };
    let app_info = vk::ApplicationInfo::default();

    let instance_info = InstanceCreateInfo::builder()
        .application_info(&app_info)
        .build();
    let instance = unsafe { entry.create_instance(&instance_info, None).unwrap() };

    let physical_device = unsafe { instance.enumerate_physical_devices().unwrap() };

    let mut device_with_queue_idx = Vec::new();

    for device in physical_device {
        let queue_family = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (idx, props) in queue_family.iter().enumerate() {
            if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                println!("device: {device:?}, idx: {idx}");
                device_with_queue_idx.push((device, idx));
            }
        }
    }

    let queue_priorities = [1.0];
    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(device_with_queue_idx[0].1 as u32)
        .queue_priorities(&queue_priorities)
        .build();

    let device_features = vk::PhysicalDeviceFeatures::default();
    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&[queue_info])
        .enabled_features(&device_features)
        .build();

    let device = unsafe {
        instance
            .create_device(device_with_queue_idx[0].0, &device_create_info, None)
            .unwrap()
    };

    let props = unsafe { instance.get_physical_device_properties(device_with_queue_idx[0].0) };
    println!("props: {:?}", &unsafe {
        ::std::ffi::CStr::from_ptr(props.device_name.as_ptr())
    });
    // let queue = unsafe { device.get_device_queue(device_with_queue_idx[0].1 as u32, 0) };

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
                // a[global_id.x] = f32(global_id.x);
            }
        ";

    let mut frontend = naga::front::wgsl::Frontend::new();
    let module = frontend.parse(src).unwrap();

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    let info = validator.validate(&module).unwrap();
    // println!("info: {module:?}");
    let mut data = Vec::new();

    let mut writer = naga::back::spv::Writer::new(&Options::default()).unwrap();
    writer
        .write(
            &module,
            &info,
            Some(&PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
            &None,
            &mut data,
        )
        .unwrap();

    let binary_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, size_of_val(data.as_slice()))
    };

    let shader_module = unsafe {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: binary_slice.len(),
            p_code: binary_slice.as_ptr() as _,
            ..Default::default()
        };
        device
            .create_shader_module(&shader_module_create_info, None)
            .unwrap()
    };

    let dispatch_size = 655360;

    pub unsafe fn create_buffer<T>(device: &ash::Device, size: usize) -> Buffer {
        let buffer_size = size * std::mem::size_of::<T>();
        let buffer_create_info = vk::BufferCreateInfo {
            size: buffer_size as vk::DeviceSize,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            ..Default::default()
        };
        device.create_buffer(&buffer_create_info, None).unwrap()
    }

    pub unsafe fn allocate_memory(
        instance: &ash::Instance,
        device: &ash::Device,
        mem_req: vk::MemoryRequirements,
        physical_device: PhysicalDevice,
    ) -> DeviceMemory {
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let memory_type_index = get_memory_type_index(
            &memory_properties,
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("no suitable memory type found");
        let memory_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index,
            ..Default::default()
        };
        device.allocate_memory(&memory_allocate_info, None).unwrap()
    }

    let buffer = unsafe { create_buffer::<f32>(&device, dispatch_size) };

    let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem = unsafe { allocate_memory(&instance, &device, mem_req, device_with_queue_idx[0].0) };
    unsafe { device.bind_buffer_memory(buffer, mem, 0).unwrap() };

    let mapping = unsafe { device.map_memory(mem, 0, vk::WHOLE_SIZE, Default::default()) }.unwrap();
    let data = unsafe { core::slice::from_raw_parts_mut(mapping as *mut f32, dispatch_size) };
    for (i, v) in data.iter_mut().enumerate() {
        *v = 4.0;
    }

    unsafe { device.unmap_memory(mem) };

    let buffer1 = unsafe { create_buffer::<f32>(&device, dispatch_size) };

    let mem_req = unsafe { device.get_buffer_memory_requirements(buffer1) };
    let mem1 = unsafe { allocate_memory(&instance, &device, mem_req, device_with_queue_idx[0].0) };
    unsafe { device.bind_buffer_memory(buffer1, mem1, 0).unwrap() };

    let mapping =
        unsafe { device.map_memory(mem1, 0, vk::WHOLE_SIZE, Default::default()) }.unwrap();
    let data = unsafe { core::slice::from_raw_parts_mut(mapping as *mut f32, dispatch_size) };
    for (i, v) in data.iter_mut().enumerate() {
        *v = 3.0;
    }

    unsafe { device.unmap_memory(mem1) };

    let buffer2 = unsafe { create_buffer::<f32>(&device, dispatch_size) };

    let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem2 = unsafe { allocate_memory(&instance, &device, mem_req, device_with_queue_idx[0].0) };

    unsafe { device.bind_buffer_memory(buffer2, mem2, 0).unwrap() }; // make the pipeline layout

    let descriptor_set_layout = {
        let descriptor_set_layout_bindings = (0..3)
            .map(|i| vk::DescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_set_layout_bindings);

        unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }
            .unwrap()
    };

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
        device.create_compute_pipelines(PipelineCache::null(), &[pipeline_create_info], None)
    }
    .unwrap()[0];

    // create a pool for the descriptor we need
    let descriptor_pool = {
        let descriptor_pool_sizes = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        };
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(core::slice::from_ref(&descriptor_pool_sizes));

        unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }.unwrap()
    };

    // allocate and write the descriptor set
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(core::slice::from_ref(&descriptor_set_layout));
    let descriptor_set =
        unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }.unwrap()[0];

    let descriptor_buffer_info = [vk::DescriptorBufferInfo {
        buffer,
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let descriptor_buffer_info1 = [vk::DescriptorBufferInfo {
        buffer: buffer1,
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let descriptor_buffer_info2 = [vk::DescriptorBufferInfo {
        buffer: buffer2,
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let write_descriptor_set = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&descriptor_buffer_info);
    let write_descriptor_set1 = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&descriptor_buffer_info1);

    let write_descriptor_set2 = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&descriptor_buffer_info2);
    let write_descriptor_sets = [
        write_descriptor_set.build(),
        write_descriptor_set1.build(),
        write_descriptor_set2.build(),
    ];
    unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };

    // run a command buffer to run the shader
    let command_pool_create_info = vk::CommandPoolCreateInfo {
        queue_family_index: device_with_queue_idx[0].1 as u32,
        ..Default::default()
    };
    let command_pool =
        unsafe { device.create_command_pool(&command_pool_create_info, None) }.unwrap();
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let command_buffer =
        unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }.unwrap()[0];

    // make a command buffer that runs the compute shader
    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }.unwrap();
    unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline) };
    unsafe {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            core::slice::from_ref(&descriptor_set),
            &[],
        )
    };
    unsafe { device.cmd_dispatch(command_buffer, 256 * 16, 1, 1) };
    unsafe { device.end_command_buffer(command_buffer) }.unwrap();

    // run it and wait until it is completed
    let queue = unsafe { device.get_device_queue(device_with_queue_idx[0].1 as u32, 0) };
    let submit_info =
        vk::SubmitInfo::builder().command_buffers(core::slice::from_ref(&command_buffer));

    const TIMES: usize = 100;
    let start = Instant::now();
    for _ in 0..TIMES {
        unsafe { device.queue_submit(queue, core::slice::from_ref(&submit_info), Fence::null()) }
            .unwrap();
        unsafe { device.device_wait_idle() }.unwrap();
        // println!("fin");
    }

    println!("elapsed: {:?}", start.elapsed() /*/ TIMES as u32 */);

    // check results
    let mapping =
        unsafe { device.map_memory(mem2, 0, vk::WHOLE_SIZE, Default::default()) }.unwrap();
    let check = unsafe { core::slice::from_raw_parts(mapping as *const f32, dispatch_size) };
    // println!("check: {:?}", check);
    for (i, v) in check.iter().copied().enumerate() {
        assert_eq!(7.0, v);
    }
    println!("compute shader run successfully!");

    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}

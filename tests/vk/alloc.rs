use std::rc::Rc;

use ash::vk::{self, BufferUsageFlags, DescriptorType, Fence, MemoryPropertyFlags};
use custos::vulkan::{
    allocate_memory, create_buffer, create_descriptor_infos, create_write_descriptor_sets, Context,
    ShaderCache, VkArray,
};

#[test]
// #[ignore = "reason"]
fn test_vk_alloc() {
    let context = Rc::new(Context::new(0).unwrap());

    let data = (0..1_000_800).map(|x| x as f32).collect::<Vec<_>>();

    let _lhs = VkArray::from_slice(
        context.clone(),
        &data,
        BufferUsageFlags::STORAGE_BUFFER,
        custos::flag::AllocFlag::None,
    )
    .unwrap();
    // println!("lhs: {:?}", &lhs[1_000_000..1_000_700]);

    let _x = VkArray::<f32>::new(
        context.clone(),
        data.len(),
        BufferUsageFlags::STORAGE_BUFFER,
        custos::flag::AllocFlag::None,
        MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .unwrap();

    let len = data.len();
    let buf = unsafe {
        create_buffer::<i32>(
            &context.device,
            BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::TRANSFER_SRC
                | BufferUsageFlags::TRANSFER_DST,
            len,
        )
        .unwrap()
    };
    let mem_req = unsafe { context.device.get_buffer_memory_requirements(buf) };

    let _mem = unsafe {
        allocate_memory(
            &context.device,
            mem_req,
            &context.memory_properties,
            MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap()
    };
}

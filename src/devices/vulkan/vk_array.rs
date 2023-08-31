use ash::{prelude::VkResult, vk};
use std::rc::Rc;

use super::context::Context;

pub struct VkArray<T> {
    len: usize,
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    context: Rc<Context>,
    mapped_ptr: *mut T,
}

impl<T> VkArray<T> {
    pub fn new(context: Rc<Context>, len: usize) -> crate::Result<Self> {
        let buf = unsafe { create_buffer::<T>(&context, len)? };
        let mem_req = unsafe { context.get_buffer_memory_requirements(buf) };

        let mem = unsafe { allocate_memory(&context, mem_req, &context.memory_properties)? };
        unsafe { context.bind_buffer_memory(buf, mem, 0)? };

        let mapped_ptr = unsafe {
            context
                .logical_device
                .map_memory(mem, 0, vk::WHOLE_SIZE, Default::default())?
        } as *mut T;

        Ok(VkArray {
            len,
            buf,
            mem,
            context,
            mapped_ptr,
        })
    }

    #[inline]
    pub fn from_slice(device: Rc<Context>, data: &[T]) -> crate::Result<Self>
    where
        T: Copy,
    {
        let mut array = VkArray::<T>::new(device, data.len())?;
        array.as_mut_slice().copy_from_slice(data);
        Ok(array)
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.mapped_ptr, self.len) }
    }
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.mapped_ptr, self.len) }
    }
}

impl<T> Drop for VkArray<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.context.unmap_memory(self.mem);
            self.context.free_memory(self.mem, None);
            self.context.destroy_buffer(self.buf, None)
        }
    }
}

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

pub unsafe fn create_buffer<T>(device: &ash::Device, size: usize) -> VkResult<vk::Buffer> {
    let buffer_size = size * std::mem::size_of::<T>();
    let buffer_create_info = vk::BufferCreateInfo {
        size: buffer_size as vk::DeviceSize,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        ..Default::default()
    };
    device.create_buffer(&buffer_create_info, None)
}

pub unsafe fn allocate_memory(
    device: &ash::Device,
    mem_req: vk::MemoryRequirements,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> VkResult<vk::DeviceMemory> {
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
    device.allocate_memory(&memory_allocate_info, None)
}

#[cfg(test)]
mod tests {
    use crate::{vulkan::vulkan_device::Vulkan, Base};

    use super::VkArray;

    #[test]
    fn test_vk_array_allocation() {
        let device = Vulkan::<Base>::new(0).unwrap();
        let arr1 = VkArray::<f32>::new(device.context(), 10).unwrap();
    }
}

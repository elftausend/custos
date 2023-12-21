use ash::{
    prelude::VkResult,
    vk::{self, BufferUsageFlags},
};
use core::ops::{Deref, DerefMut};
use std::rc::Rc;

use crate::{flag::AllocFlag, HasId, HostPtr, PtrType, ShallowCopy};

use super::context::Context;

pub struct VkArray<T> {
    pub len: usize,
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub context: Rc<Context>,
    pub mapped_ptr: *mut T,
    pub flag: AllocFlag,
}

impl<T> PtrType for VkArray<T> {
    #[inline]
    fn size(&self) -> usize {
        self.len
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        self.flag
    }

    unsafe fn set_flag(&mut self, flag: AllocFlag) {
        self.flag = flag;
    }
}

impl<T> HasId for VkArray<T> {
    #[inline]
    fn id(&self) -> crate::Id {
        crate::Id {
            id: self.mapped_ptr as u64,
            len: self.len,
        }
    }
}

impl<T> VkArray<T> {
    pub fn new(
        context: Rc<Context>,
        len: usize,
        usage_flag: BufferUsageFlags,
        flag: AllocFlag,
    ) -> crate::Result<Self> {
        let buf = unsafe { create_buffer::<T>(&context.device, usage_flag, len)? };
        let mem_req = unsafe { context.device.get_buffer_memory_requirements(buf) };

        let mem = unsafe { allocate_memory(&context.device, mem_req, &context.memory_properties)? };
        unsafe { context.device.bind_buffer_memory(buf, mem, 0)? };

        let mapped_ptr = unsafe {
            context
                .device
                .map_memory(mem, 0, vk::WHOLE_SIZE, Default::default())?
        } as *mut T;

        Ok(VkArray {
            len,
            buf,
            mem,
            context,
            mapped_ptr,
            flag,
        })
    }

    #[inline]
    pub fn from_slice(
        context: Rc<Context>,
        data: &[T],
        usage_flag: BufferUsageFlags,
        flag: AllocFlag,
    ) -> crate::Result<Self>
    where
        T: Clone,
    {
        let mut array = VkArray::<T>::new(context, data.len(), usage_flag, flag)?;
        array.clone_from_slice(data);
        Ok(array)
    }
}

impl<T> ShallowCopy for VkArray<T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        VkArray {
            len: self.len,
            buf: self.buf,
            mem: self.mem,
            context: self.context.clone(),
            mapped_ptr: self.mapped_ptr,
            flag: AllocFlag::Wrapper,
        }
    }
}

impl<T> Drop for VkArray<T> {
    #[inline]
    fn drop(&mut self) {
        if !self.flag.continue_deallocation() {
            return;
        }
        unsafe {
            self.context.device.unmap_memory(self.mem);
            self.context.device.free_memory(self.mem, None);
            self.context.device.destroy_buffer(self.buf, None)
        }
    }
}

impl<T> HostPtr<T> for VkArray<T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.mapped_ptr
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.mapped_ptr
    }
}

impl<T> Deref for VkArray<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr(), self.size()) }
    }
}

impl<T> DerefMut for VkArray<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr_mut(), self.size()) }
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

pub unsafe fn create_buffer<T>(
    device: &ash::Device,
    usage: BufferUsageFlags,
    size: usize,
) -> VkResult<vk::Buffer> {
    let buffer_size = size * std::mem::size_of::<T>();
    let buffer_create_info = vk::BufferCreateInfo {
        size: buffer_size as vk::DeviceSize,
        usage,
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
        memory_properties,
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
    use super::VkArray;
    use crate::{vulkan::context::Context, HostPtr};
    use ash::vk::BufferUsageFlags;
    use std::rc::Rc;

    #[test]
    fn test_vk_array_allocation() {
        let context = Rc::new(Context::new(0).unwrap());
        let _arr1 = VkArray::<f32>::new(
            context.clone(),
            10,
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();
    }

    #[test]
    fn test_vk_array_from_slice() {
        let context = Rc::new(Context::new(0).unwrap());
        let arr1 = VkArray::<f32>::from_slice(
            context.clone(),
            &[1., 2., 3., 4., 5., 6.],
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .unwrap();
        assert_eq!(arr1.as_slice(), &[1., 2., 3., 4., 5., 6.,])
    }
}

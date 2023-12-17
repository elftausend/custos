use ash::vk::BufferUsageFlags;

use crate::{
    flag::AllocFlag,
    prelude::Number,
    vulkan::{Context, VkArray},
    OnDropBuffer, Shape, Vulkan,
};
use std::rc::Rc;

pub struct VkShaderArgument {
    // a number allocates this array, which is dealloated after usage
    pub vk_array_handle: Option<VkArray<u8>>,
    pub buffer: ash::vk::Buffer,
    pub descriptor_type: ash::vk::DescriptorType,
}

pub trait AsVkShaderArgument {
    fn as_arg(&self, context: Rc<Context>) -> VkShaderArgument;
}

impl<T: Number> AsVkShaderArgument for T {
    fn as_arg(&self, context: Rc<Context>) -> VkShaderArgument {
        let vk_array_handle = VkArray::from_slice(
            context,
            &[*self],
            BufferUsageFlags::STORAGE_BUFFER,
            AllocFlag::None,
        )
        .unwrap();
        let buffer = vk_array_handle.buf;
        let vk_array_handle =
            Some(unsafe { std::mem::transmute::<_, VkArray<u8>>(vk_array_handle) });

        VkShaderArgument {
            vk_array_handle,
            buffer,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

impl<T> AsVkShaderArgument for VkArray<T> {
    #[inline]
    fn as_arg(&self, _context: Rc<Context>) -> VkShaderArgument {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: self.buf,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

impl AsVkShaderArgument for ash::vk::Buffer {
    #[inline]
    fn as_arg(&self, _context: Rc<Context>) -> VkShaderArgument {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: *self,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

impl AsVkShaderArgument for &ash::vk::Buffer {
    #[inline]
    fn as_arg(&self, _context: Rc<Context>) -> VkShaderArgument {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: **self,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

impl<T> AsVkShaderArgument for &VkArray<T> {
    #[inline]
    fn as_arg(&self, _context: Rc<Context>) -> VkShaderArgument {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: self.buf,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

impl<'a, T, S, Mods> AsVkShaderArgument for crate::Buffer<'a, T, Vulkan<Mods>, S>
where
    S: Shape,
    Mods: OnDropBuffer,
{
    #[inline]
    fn as_arg(&self, _context: Rc<Context>) -> VkShaderArgument {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: self.base().buf,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

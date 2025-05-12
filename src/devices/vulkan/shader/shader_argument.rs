use ash::vk::BufferUsageFlags;

use crate::{
    Shape, Unit, Vulkan, WrappedData,
    flag::AllocFlag,
    vulkan::{Context, VkArray},
    wgsl::{AsShaderArg, WgslNumber, WgslShaderLaunch},
};
use core::marker::PhantomData;
use std::rc::Rc;

pub struct VkShaderArgument<'a> {
    // a number allocates this array, which is dealloated after usage
    pub vk_array_handle: Option<VkArray<u8>>,
    pub buffer: ash::vk::Buffer,
    pub descriptor_type: ash::vk::DescriptorType,
    pd: PhantomData<&'a u8>,
}

pub trait AsVkShaderArgument {
    fn as_arg<'a>(&self, context: Rc<Context>) -> VkShaderArgument;
}

impl<T: WgslNumber> AsVkShaderArgument for T {
    fn as_arg<'a>(&'a self, context: Rc<Context>) -> VkShaderArgument<'a> {
        let vk_array_handle = VkArray::from_slice(
            context,
            &[*self],
            BufferUsageFlags::UNIFORM_BUFFER,
            AllocFlag::None,
        )
        .unwrap();
        let buffer = vk_array_handle.buf;
        let vk_array_handle =
            Some(unsafe { std::mem::transmute::<_, VkArray<u8>>(vk_array_handle) });

        VkShaderArgument {
            vk_array_handle,
            buffer,
            descriptor_type: ash::vk::DescriptorType::UNIFORM_BUFFER,
            pd: PhantomData,
        }
    }
}

impl<T> AsVkShaderArgument for VkArray<T> {
    #[inline]
    fn as_arg<'a>(&'a self, _context: Rc<Context>) -> VkShaderArgument<'a> {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: self.buf,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
            pd: PhantomData,
        }
    }
}

impl AsVkShaderArgument for ash::vk::Buffer {
    #[inline]
    fn as_arg<'a>(&'a self, _context: Rc<Context>) -> VkShaderArgument<'a> {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: *self,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
            pd: PhantomData,
        }
    }
}

impl AsVkShaderArgument for &ash::vk::Buffer {
    #[inline]
    fn as_arg<'a>(&'a self, _context: Rc<Context>) -> VkShaderArgument<'a> {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: **self,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
            pd: PhantomData,
        }
    }
}

impl<T> AsVkShaderArgument for &VkArray<T> {
    #[inline]
    fn as_arg<'a>(&'a self, _context: Rc<Context>) -> VkShaderArgument<'a> {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: self.buf,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
            pd: PhantomData,
        }
    }
}

impl<'a, T, S, Mods> AsVkShaderArgument for crate::Buffer<'a, T, Vulkan<Mods>, S>
where
    T: Unit,
    S: Shape,
    Mods: WrappedData,
{
    #[inline]
    fn as_arg(&self, _context: Rc<Context>) -> VkShaderArgument {
        VkShaderArgument {
            vk_array_handle: None,
            buffer: self.base().buf,
            descriptor_type: ash::vk::DescriptorType::STORAGE_BUFFER,
            pd: PhantomData,
        }
    }
}

impl<T: AsVkShaderArgument + 'static> AsShaderArg<Vulkan> for T {
    #[inline]
    fn arg(&self) -> &<Vulkan as WgslShaderLaunch>::ShaderArg {
        self
    }

    #[inline]
    fn arg_mut(&mut self) -> &mut <Vulkan as WgslShaderLaunch>::ShaderArg {
        self
    }
}

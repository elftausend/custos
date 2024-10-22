use ash::{
    prelude::VkResult,
    vk::{self, BufferUsageFlags},
};
use core::{
    mem::size_of,
    ops::{Deref, DerefMut},
    ptr::null_mut,
};
use std::rc::Rc;

use crate::{flag::AllocFlag, HasId, HostPtr, PtrType, ShallowCopy, Unit, WrappedCopy};

use super::{context::Context, submit_and_wait};

pub struct VkArray<T> {
    pub len: usize,
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub context: Rc<Context>,
    pub mapped_ptr: *mut T,
    pub flag: AllocFlag,
}

unsafe impl<T: Sync> Sync for VkArray<T> {}
unsafe impl<T: Send> Send for VkArray<T> {}

impl<T: Unit> PtrType for VkArray<T> {
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

impl<T: Unit> VkArray<T> {
    pub fn new(
        context: Rc<Context>,
        len: usize,
        usage_flag: BufferUsageFlags,
        flag: AllocFlag,
        mem_prop_flags: vk::MemoryPropertyFlags,
    ) -> crate::Result<Self> {
        let buf = unsafe { create_buffer::<T>(&context.device, usage_flag, len)? };
        let mem_req = unsafe { context.device.get_buffer_memory_requirements(buf) };

        let mem = unsafe {
            allocate_memory(
                &context.device,
                mem_req,
                &context.memory_properties,
                mem_prop_flags,
            )?
        };
        unsafe { context.device.bind_buffer_memory(buf, mem, 0)? };

        let mapped_ptr = if vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            == mem_prop_flags
        {
            unsafe {
                context
                    .device
                    .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    as *mut T
            }
        } else {
            null_mut()
        };

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
    pub fn len(&self) -> usize {
        self.len
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
        let mut array = VkArray::<T>::new(
            context,
            data.len(),
            usage_flag,
            flag,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        array.clone_from_slice(data);
        Ok(array)
    }

    pub fn write_staged(&self, data: &[T])
    where
        T: Clone,
    {
        let mut src_buffer = VkArray::<T>::new(
            self.context.clone(),
            self.len(),
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
            crate::flag::AllocFlag::None,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

        src_buffer.deref_mut().clone_from_slice(data);

        self.write_buf(&src_buffer)
    }

    pub fn write_buf(&self, src_buf: &VkArray<T>)
    where
        T: Clone,
    {
        let ctx = &self.context;
        let device = &ctx.device;
        let command_buffer = ctx.command_buffer;

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }.unwrap();

        let buffer_copy_region = vk::BufferCopy::default()
            .dst_offset(0)
            .src_offset(0)
            .size((self.len() * size_of::<T>()) as u64);

        unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                src_buf.buf,
                self.buf,
                core::slice::from_ref(&buffer_copy_region),
            )
        };

        unsafe { device.end_command_buffer(command_buffer).unwrap() };
        submit_and_wait(device, command_buffer, ctx.compute_family_idx as u32).unwrap();
    }

    pub fn read_staged(&self) -> VkArray<T>
    where
        T: Clone,
    {
        let ctx = &self.context;
        let device = &ctx.device;
        let command_buffer = ctx.command_buffer;

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        let dst_buffer = VkArray::<T>::new(
            ctx.clone(),
            self.len(),
            BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::TRANSFER_DST
                | BufferUsageFlags::TRANSFER_SRC,
            crate::flag::AllocFlag::None,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

        unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }.unwrap();

        let buffer_copy_region = vk::BufferCopy::default()
            .dst_offset(0)
            .src_offset(0)
            .size((self.len() * size_of::<T>()) as u64);

        unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                self.buf,
                dst_buffer.buf,
                core::slice::from_ref(&buffer_copy_region),
            )
        };

        unsafe { device.end_command_buffer(command_buffer).unwrap() };
        submit_and_wait(device, command_buffer, ctx.compute_family_idx as u32).unwrap();

        dst_buffer
    }

    #[inline]
    pub fn read_staged_to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.read_staged().to_vec()
    }
}

impl<T> WrappedCopy for VkArray<T> {
    type Base = Self;

    #[inline]
    fn wrapped_copy(&self, to_wrap: Self::Base) -> Self {
        to_wrap
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
            if !self.mapped_ptr.is_null() {
                self.context.device.unmap_memory(self.mem);
            }
            self.context.device.free_memory(self.mem, None);
            self.context.device.destroy_buffer(self.buf, None)
        }
    }
}

impl<T: Unit> HostPtr<T> for VkArray<T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.mapped_ptr
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.mapped_ptr
    }
}

// TODO: impl deref only when using unified memory
impl<T: Unit> Deref for VkArray<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        assert!(!self.ptr().is_null());
        unsafe { std::slice::from_raw_parts(self.ptr(), self.size()) }
    }
}

impl<T: Unit> DerefMut for VkArray<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        assert!(!self.ptr_mut().is_null());
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
    let buffer_create_info = vk::BufferCreateInfo::default()
        .size(buffer_size as u64)
        .usage(usage);
    device.create_buffer(&buffer_create_info, None)
}

pub unsafe fn allocate_memory(
    device: &ash::Device,
    mem_req: vk::MemoryRequirements,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    mem_prop_flags: vk::MemoryPropertyFlags,
) -> VkResult<vk::DeviceMemory> {
    let memory_type_index =
        get_memory_type_index(memory_properties, mem_req.memory_type_bits, mem_prop_flags)
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
    use crate::vulkan::context::Context;
    use ash::vk::{self, BufferUsageFlags};
    use core::ops::Deref;
    use std::rc::Rc;

    #[test]
    fn test_vk_array_allocation() {
        let context = Rc::new(Context::new(0).unwrap());
        let _arr1 = VkArray::<f32>::new(
            context.clone(),
            10,
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
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
        assert_eq!(arr1.deref(), &[1., 2., 3., 4., 5., 6.,])
    }

    #[test]
    fn test_vk_array_write_read_staged() {
        let context = Rc::new(Context::new(0).unwrap());
        let arr1 = VkArray::<f32>::new(
            context.clone(),
            9,
            BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::TRANSFER_DST
                | BufferUsageFlags::TRANSFER_SRC,
            crate::flag::AllocFlag::None,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap();

        arr1.write_staged(&[1., 3., 5., 7., 9., 11., 13., 15., 17.]);

        let out = arr1.read_staged_to_vec();
        assert_eq!(out, [1., 3., 5., 7., 9., 11., 13., 15., 17.])
    }
}

use ash::{
    vk::{
        self, CommandBuffer, InstanceCreateInfo, PhysicalDevice, PhysicalDeviceMemoryProperties,
        PhysicalDeviceProperties,
    },
    Device, Entry,
};

use crate::{Base, Module, Setup};

use super::list_compute_devices;

pub struct Vulkan<Mods = Base> {
    modules: Mods,
    physical_device: PhysicalDevice,
    compute_family_idx: usize,
    logical_device: Device,
    device_props: PhysicalDeviceProperties,
    command_buffer: CommandBuffer,
    memory_properties: PhysicalDeviceMemoryProperties,
}

impl<SimpleMods> Vulkan<SimpleMods> {
    #[inline]
    pub fn new<NewMods>(idx: usize) -> crate::Result<Vulkan<NewMods>>
    where
        SimpleMods: Module<Vulkan, Module = NewMods>,
        NewMods: Setup<Vulkan<NewMods>>,
    {
        let entry = unsafe { Entry::load()? };
        let app_info = vk::ApplicationInfo::default();

        let instance_info = InstanceCreateInfo::builder()
            .application_info(&app_info)
            .build();
        let instance = unsafe { entry.create_instance(&instance_info, None).unwrap() };

        let (physical_device, compute_family_idx) = list_compute_devices(&instance)[idx];

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(compute_family_idx as u32)
            .queue_priorities(&queue_priorities)
            .build();

        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[queue_info])
            .enabled_features(&device_features)
            .build();

        let logical_device =
            unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };

        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index: compute_family_idx as u32,
            ..Default::default()
        };
        let command_pool =
            unsafe { logical_device.create_command_pool(&command_pool_create_info, None) }?;
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let command_buffer =
            unsafe { logical_device.allocate_command_buffers(&command_buffer_allocate_info) }?[0];

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let mut vulkan = Vulkan {
            modules: SimpleMods::new(),
            physical_device,
            compute_family_idx,
            logical_device,
            device_props,
            command_buffer,
            memory_properties,
        };
        NewMods::setup(&mut vulkan);
        Ok(vulkan)
    }
}

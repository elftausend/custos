use core::ffi::{c_char, CStr};

use ash::{
    vk::{
        self, CommandBuffer, InstanceCreateInfo, PhysicalDevice, PhysicalDeviceMemoryProperties,
        PhysicalDeviceProperties, PipelineCache,
    },
    Device, Entry,
};

use super::{
    list_compute_devices,
    shader::{create_command_buffer, create_command_pool},
};

pub struct Context {
    pub physical_device: PhysicalDevice,
    pub compute_family_idx: usize,
    pub device: Device,
    pub device_props: PhysicalDeviceProperties,
    pub command_buffer: CommandBuffer,
    pub memory_properties: PhysicalDeviceMemoryProperties,
    pub pipeline_cache: PipelineCache,
    pub entry: Entry,
}

impl Context {
    #[inline]
    pub fn new(device_idx: usize) -> crate::Result<Self> {
        let entry = unsafe { Entry::load()? };
        let app_info = vk::ApplicationInfo::default();

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let instance_info = InstanceCreateInfo::builder()
            .enabled_layer_names(&layers_names_raw)
            .application_info(&app_info)
            .build();
        let instance = unsafe { entry.create_instance(&instance_info, None).unwrap() };
        let (physical_device, compute_family_idx) = list_compute_devices(&instance)[device_idx];

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

        let command_pool = create_command_pool(&logical_device, compute_family_idx)?;
        let command_buffer = create_command_buffer(&logical_device, command_pool)?;

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Ok(Context {
            physical_device,
            compute_family_idx,
            device: logical_device,
            device_props,
            command_buffer,
            memory_properties,
            entry,
            pipeline_cache: PipelineCache::default(),
        })
    }
}

impl Drop for Context {
    #[inline]
    fn drop(&mut self) {
        unsafe {}
    }
}

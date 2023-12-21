use core::ffi::{c_char, CStr};

use ash::{
    vk::{
        self, CommandBuffer, CommandPool, InstanceCreateInfo, PhysicalDevice,
        PhysicalDeviceMemoryProperties, PhysicalDeviceProperties, PipelineCache,
    },
    Device, Entry, Instance,
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
    pub command_pool: CommandPool,
    pub instance: Instance,
    pub memory_properties: PhysicalDeviceMemoryProperties,
    pub pipeline_cache: PipelineCache,
    pub entry: Entry,
}

pub fn validation_layers<'a>() -> Vec<&'a CStr> {
    let use_validation_layer = std::env::var("CUSTOS_VK_USE_VALIDATION_LAYER")
        .unwrap_or("false".into())
        .parse::<bool>()
        .unwrap_or_default();
    if use_validation_layer {
        unsafe {
            vec![CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        }
    } else {
        vec![]
    }
}

impl Context {
    #[inline]
    pub fn new(device_idx: usize) -> crate::Result<Self> {
        let entry = unsafe { Entry::load()? };
        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"custos\0") };
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let layer_names = validation_layers();

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
            .queue_create_infos(std::slice::from_ref(&queue_info))
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
            command_pool,
            memory_properties,
            entry,
            instance,
            pipeline_cache: PipelineCache::default(),
        })
    }
}

impl Drop for Context {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);

            self.instance.destroy_instance(None);
        }
    }
}

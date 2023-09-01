use ash::{
    vk::{self, PhysicalDevice},
    Instance,
};

mod context;
mod shader;
mod vk_array;
mod vulkan_device;

pub fn list_compute_devices(instance: &Instance) -> Vec<(PhysicalDevice, usize)> {
    let physical_device = unsafe { instance.enumerate_physical_devices().unwrap() };

    let mut physical_dev_with_queue_idx = Vec::new();

    for device in physical_device {
        let queue_family = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (idx, props) in queue_family.iter().enumerate() {
            if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                physical_dev_with_queue_idx.push((device, idx));
            }
        }
    }
    physical_dev_with_queue_idx
}

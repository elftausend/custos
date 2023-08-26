
use ash::{vk::{self, InstanceCreateInfo, Instance}, Entry};

#[test]
fn test_vulkan_compute_with_wgsl_and_spirv() {
    unsafe {
        let entry = Entry::load().unwrap();
        let app_info = vk::ApplicationInfo::default();

        let instance_info = InstanceCreateInfo::builder().application_info(&app_info).build();
        let instance = entry.create_instance(&instance_info, None).unwrap();

        let physical_device = instance.enumerate_physical_devices().unwrap();

        let mut device_with_queue_idx = Vec::new();

        for device in physical_device {
            let queue_family = instance.get_physical_device_queue_family_properties(device);
            for (idx, props) in queue_family.iter().enumerate() {
                if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    println!("device: {device:?}, idx: {idx}");
                    device_with_queue_idx.push((device, idx));
                }
            }
        }

    }

    
        
}
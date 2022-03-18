
use super::{api::{DeviceType, get_device_ids, get_platforms, OCLError, OCLErrorKind}, CL_CACHE, cl_device::CLDevice};

#[derive(Debug)]
pub struct CLDevices {
    pub current_devices: Vec<CLDevice>,
}

impl CLDevices {
    pub fn get_current(&self, device_idx: usize) -> Result<CLDevice, OCLError> {
        if device_idx < self.current_devices.len() {
            Ok(self.current_devices[device_idx])
        } else {
            Err(OCLError::with_kind(OCLErrorKind::InvalidDeviceIdx))
        }
    }
}


lazy_static::lazy_static! {
    pub static ref CL_DEVICES: CLDevices = {
        let mut current_devices = Vec::new();
        unsafe {
            CL_CACHE.sync()
        }
        
        let platform = get_platforms().unwrap()[0];
        let devices = get_device_ids(platform, &(DeviceType::GPU as u64)).unwrap();
    
        for device in devices {
            current_devices.push(CLDevice::new(device).unwrap())
        }
        CLDevices { current_devices }
    };
}
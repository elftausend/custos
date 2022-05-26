
use crate::Error;

use super::{api::{DeviceType, get_device_ids, get_platforms, OCLErrorKind}, cl_device::CLDevice};


lazy_static::lazy_static! {
    pub static ref CL_DEVICES: CLDevices = CLDevices::new();
}

#[derive(Debug)]
pub struct CLDevices {
    pub current_devices: Vec<CLDevice>,
}

impl CLDevices {
    fn new() -> CLDevices {
        let mut current_devices = Vec::new();
        
        let platform = get_platforms().unwrap()[0];
        let devices = get_device_ids(platform, &(DeviceType::GPU as u64)).unwrap();
    
        for device in devices {
            current_devices.push(CLDevice::new(device).unwrap())
        }
        CLDevices { current_devices }
    }

    pub fn current(&self, device_idx: usize) -> Result<CLDevice, Error> {
        if device_idx < self.current_devices.len() {
            Ok(self.current_devices[device_idx].clone())
        } else {
            Err(Error::from(OCLErrorKind::InvalidDeviceIdx))
        }
    }
}

impl Default for CLDevices {
    fn default() -> Self {
        Self::new()
    }
}

use crate::Error;
use super::{api::{DeviceType, get_device_ids, get_platforms, OCLErrorKind}, cl_device::CLDevice};

lazy_static::lazy_static! {
    pub static ref CL_DEVICES: CLDevices = CLDevices::new().unwrap();
}

#[derive(Debug)]
pub struct CLDevices {
    pub current_devices: Vec<CLDevice>,
}

impl CLDevices {
    fn new() -> Result<CLDevices, crate::Error> {
        let mut current_devices = Vec::new();
        
        let platform = get_platforms()?[0];
        let devices = get_device_ids(platform, &(DeviceType::GPU as u64))?;
    
        for device in devices {
            current_devices.push(CLDevice::new(device)?)
        }
        Ok(CLDevices { current_devices })
    }

    pub fn current(&self, device_idx: usize) -> Result<CLDevice, Error> {
        if device_idx >= self.current_devices.len() {
            return Err(OCLErrorKind::InvalidDeviceIdx.into());
        } 
        Ok(self.current_devices[device_idx].clone())
        
    }
}

impl Default for CLDevices {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
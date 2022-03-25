
use super::{api::{DeviceType, get_device_ids, get_platforms, OCLError, OCLErrorKind}, cl_device::CLDevice};

/*
lazy_static::lazy_static! {
    pub static ref CL_DEVICES: CLDevices = CLDevices::new();
}
*/

lazy_static::lazy_static! {
    pub static ref CL_DEVICES2: CLDevices2 = CLDevices2::new();
}

#[derive(Debug)]
pub struct CLDevices2 {
    pub current_devices: Vec<CLDevice>,
}

impl CLDevices2 {
    pub fn new() -> CLDevices2 {
        let mut current_devices = Vec::new();
        
        let platform = get_platforms().unwrap()[0];
        let devices = get_device_ids(platform, &(DeviceType::GPU as u64)).unwrap();
    
        for device in devices {
            current_devices.push(CLDevice::new(device).unwrap())
        }
        CLDevices2 { current_devices }
    }

    pub fn get_current(&self, device_idx: usize) -> Result<CLDevice, OCLError> {
        if device_idx < self.current_devices.len() {
            Ok(self.current_devices[device_idx].clone())
        } else {
            Err(OCLError::with_kind(OCLErrorKind::InvalidDeviceIdx))
        }
    }
}

/* 

#[derive(Debug)]
pub struct CLDevices {
    pub current_devices: Vec<CLDevice>,
}

impl CLDevices {
    pub fn new() -> CLDevices {
        let mut current_devices = Vec::new();
        
        let platform = get_platforms().unwrap()[0];
        let devices = get_device_ids(platform, &(DeviceType::GPU as u64)).unwrap();
    
        for device in devices {
            current_devices.push(CLDevice::new(device).unwrap())
        }
        CLDevices { current_devices }
    }

    pub fn get_current(&self, device_idx: usize) -> Result<CLDevice, OCLError> {
        if device_idx < self.current_devices.len() {
            Ok(self.current_devices[device_idx])
        } else {
            Err(OCLError::with_kind(OCLErrorKind::InvalidDeviceIdx))
        }
    }
}

impl Default for CLDevices {
    fn default() -> Self {
        Self { current_devices: Default::default() }
    }
}
*/
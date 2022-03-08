use super::{cl_device::CLDevice, api::{OCLError, OCLErrorKind, get_platforms, get_device_ids, DeviceType}};

pub static mut CL_DEVICES: CLDevices = CLDevices {current_devices: Vec::new()};

#[derive(Debug)]
pub struct CLDevices {
    pub current_devices: Vec<CLDevice>,
}

impl CLDevices {
    pub fn get_current(&mut self, device_idx: usize) -> Result<&mut CLDevice, OCLError> {
        self.sync_current()?;
        
        if device_idx < self.current_devices.len() {
            Ok(&mut self.current_devices[device_idx])    
        } else {
            Err(OCLError::with_kind(OCLErrorKind::InvalidDeviceIdx))
        }
        //&mut self.current_devices.as_mut().unwrap()[device_idx]
    }
    pub fn sync_current(&mut self) -> Result<(), OCLError>{
        if self.current_devices.len() == 0 {

            let platform = get_platforms()?[0];
            let devices = get_device_ids(platform, &(DeviceType::GPU as u64))?;
            
            for device in devices {
                let cl_device = CLDevice::new(device)?;
                self.current_devices.push(cl_device)
            }
            
            
        }
        Ok(())
    }
}
use super::{cl_device::CLDevice, api::{OCLError, OCLErrorKind, get_platforms, get_device_ids, DeviceType, create_context, create_command_queue}};

pub static mut DEVICES: Devices = Devices {current_devices: Vec::new()};

#[derive(Debug)]
pub struct Devices {
    pub current_devices: Vec<CLDevice>,
}

impl Devices {
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
        if self.current_devices.len() > 0 {

            let platform = get_platforms()?[0];
            let devices = get_device_ids(platform, &(DeviceType::GPU as u64))?;
            let mut cl_devices = Vec::new();
            
            for device in devices {
                let cl_device = CLDevice::new(device)?;
                cl_devices.push(cl_device);
            }
            self.current_devices = cl_devices;
            
        }
        Ok(())
    }
}
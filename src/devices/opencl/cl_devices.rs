use super::api::{get_device_ids, get_platforms, DeviceType, OCLErrorKind};
use crate::{Error, InternCLDevice};

lazy_static::lazy_static! {
    pub static ref CL_DEVICES: CLDevices = CLDevices::new().unwrap();
}

pub struct CLDevices {
    pub current_devices: Vec<InternCLDevice>,
}

impl CLDevices {
    fn new() -> Result<CLDevices, crate::Error> {
        let mut current_devices = Vec::new();

        let platform = get_platforms()?[0];
        let devices = get_device_ids(platform, &(DeviceType::GPU as u64))?;

        for device in devices {
            current_devices.push(InternCLDevice::new(device)?)
        }
        Ok(CLDevices { current_devices })
    }

    pub fn current(&self, device_idx: usize) -> Result<InternCLDevice, Error> {
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

#[cfg(test)]
mod tests {
    use crate::opencl::CLDevices;

    #[test]
    fn test_cl_devices_creation() {
        let devices = CLDevices::default();
        assert!(devices.current_devices.len() >= 1);
    }

    #[test]
    #[should_panic]
    fn test_invalid_idx() {
        let device = CLDevices {
            current_devices: vec![],
        };
        device.current(1).unwrap();
    }
}

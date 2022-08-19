use super::api::MetalIntDevice;

pub struct MTLDevice {
    pub device: MetalIntDevice,
}

impl MTLDevice {
    pub fn new() -> Self {
        MTLDevice {
            device: MetalIntDevice::system_default().unwrap()
        }
    }
}
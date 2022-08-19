
pub enum MTLDevicePtr {}
pub enum MTLCommandQueue {}
pub enum MTLCommandBuffer {}

#[link(name = "Metal", kind = "framework")]
extern "C" {
    pub fn MTLCreateSystemDefaultDevice() -> *mut MTLDevicePtr;
}

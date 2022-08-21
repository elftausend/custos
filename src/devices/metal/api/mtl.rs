use objc::{msg_send, Message, sel, sel_impl};
use super::{MTLDevicePtr, MTLCreateSystemDefaultDevice, MTLCommandQueue, MTLCommandBuffer};

pub struct MetalIntDevice(*mut MTLDevicePtr);
unsafe impl Message for MTLDevicePtr {}

pub struct CommandQueue(*mut MTLCommandQueue);
unsafe impl Message for MTLCommandQueue {}

impl MetalIntDevice {
    pub fn system_default() -> Option<Self> {
        unsafe { MTLCreateSystemDefaultDevice().as_mut().map(|x| Self(x)) }
    }
    pub fn has_unified_memory(&self) -> bool {
        unsafe { msg_send![self.0, hasUnifiedMemory] }
    }
    
    pub fn new_command_queue(&self) -> CommandQueue {
        unsafe { msg_send![self.0, newCommandQueue] }
    }
}

pub struct CommandBuffer(*mut MTLCommandBuffer);
unsafe impl Message for MTLCommandBuffer {}

impl CommandQueue {
    pub fn new_command_buffer(&self) -> &CommandBuffer {
        unsafe { msg_send![self.0, commandBuffer] }
    }
}



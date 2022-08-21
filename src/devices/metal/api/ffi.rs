use std::ffi::c_void;

use objc::{class, msg_send, sel, sel_impl, runtime::Object};


pub fn nsstring_from_str(string: &str) -> *mut Object {
    const UTF8_ENCODING: usize = 4;

    let cls = class!(NSString);
    let bytes = string.as_ptr() as *const c_void;
    unsafe {
        let obj: *mut Object = msg_send![cls, alloc];
        let obj: *mut Object = msg_send![
            obj,
            initWithBytes:bytes
            length:string.len()
            encoding:UTF8_ENCODING
        ];
        let _: *mut c_void = msg_send![obj, autorelease];
        obj
    }
}

pub enum MTLDevicePtr {}
pub enum MTLCommandQueue {}
pub enum MTLCommandBuffer {}

#[link(name = "Metal", kind = "framework")]
extern "C" {
    pub fn MTLCreateSystemDefaultDevice() -> *mut MTLDevicePtr;
}

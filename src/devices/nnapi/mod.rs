mod nnapi_device;
pub use nnapi_device::*;

pub use nnapi::*;

/*pub fn log(priority: ndk_sys::android_LogPriority, msg: &str) {
    let tag = std::ffi::CString::new("MyApp").unwrap();
    let msg = std::ffi::CString::new(msg).unwrap();
    unsafe { ndk_sys::__android_log_print(priority.0 as i32, tag.as_ptr(), msg.as_ptr()) };
}*/

mod nnapi_device;
mod nnapi_ptr;

pub use nnapi_ptr::*;
pub use nnapi_device::*;
pub use nnapi::*;

use crate::{flag::AllocFlag, HasId, PtrConv, PtrType, Shape, OnDropBuffer};

/*pub fn log(priority: ndk_sys::android_LogPriority, msg: &str) {
    let tag = std::ffi::CString::new("MyApp").unwrap();
    let msg = std::ffi::CString::new(msg).unwrap();
    unsafe { ndk_sys::__android_log_print(priority.0 as i32, tag.as_ptr(), msg.as_ptr()) };
}*/

impl<U, Mods: OnDropBuffer> PtrConv for NnapiDevice<U, Mods> {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Data<T, IS>,
        flag: crate::flag::AllocFlag,
    ) -> Self::Data<Conv, OS> {
        NnapiPtr {
            dtype: ptr.dtype.clone(),
            idx: ptr.idx,
            flag,
        }
    }
}

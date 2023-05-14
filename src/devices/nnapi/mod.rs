mod nnapi_device;
pub use nnapi_device::*;

pub use nnapi::*;

use crate::{flag::AllocFlag, PtrConv, Shape, PtrType};

/*pub fn log(priority: ndk_sys::android_LogPriority, msg: &str) {
    let tag = std::ffi::CString::new("MyApp").unwrap();
    let msg = std::ffi::CString::new(msg).unwrap();
    unsafe { ndk_sys::__android_log_print(priority.0 as i32, tag.as_ptr(), msg.as_ptr()) };
}*/

pub struct NnapiPtr {
    dtype: Operand,
    pub idx: u32,
    flag: AllocFlag,
}

impl Default for NnapiPtr {
    fn default() -> Self {
        Self {
            dtype: Operand::activation(),
            idx: u32::MAX,
            flag: AllocFlag::Wrapper,
        }
    }
}

impl PtrConv for NnapiDevice {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Ptr<T, IS>,
        flag: crate::flag::AllocFlag,
    ) -> Self::Ptr<Conv, OS> {
        NnapiPtr {
            dtype: ptr.dtype.clone(),
            idx: ptr.idx,
            flag,
        }
    }
}

impl PtrType for NnapiPtr {
    #[inline]
    fn size(&self) -> usize {
        self.dtype.len
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        self.flag
    }
}

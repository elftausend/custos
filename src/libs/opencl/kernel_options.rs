use std::{mem::size_of, ffi::c_void};

use crate::{CLDevice, number::Number, Buffer};

use super::{CL_CACHE, api::{OCLErrorKind, set_kernel_arg_both, enqueue_nd_range_kernel}};

pub trait AsClCvoidPtr {
    fn as_cvoid_ptr(&self) -> *const c_void;
    fn is_num(&self) -> bool {
        false
    }
    fn size(&self) -> usize {
        std::mem::size_of::<*const c_void>()
    }
}

impl<T> AsClCvoidPtr for &Buffer<T> {
    fn as_cvoid_ptr(&self) -> *const c_void {
        self.ptr.1
    }
}

impl<T> AsClCvoidPtr for Buffer<T> {
    fn as_cvoid_ptr(&self) -> *const c_void {
        self.ptr.1
    }
}

impl<T: Number> AsClCvoidPtr for T {
    fn as_cvoid_ptr(&self) -> *const c_void {
        self as *const T as *const c_void
    }

    fn size(&self) -> usize {
        size_of::<T>()
    }

    fn is_num(&self) -> bool {
        true
    }
}

pub fn enqueue_kernel(
    device: &CLDevice,
    src: &str,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    args: &[&dyn AsClCvoidPtr],
) -> crate::Result<()> {
    let kernel = CL_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .arg_kernel_cache(device, src.to_string())
    })?;

    let wd;
    if gws[0] == 0 {
        return Err(OCLErrorKind::InvalidGlobalWorkSize.into());
    } else if gws[1] == 0 {
        wd = 1;
    } else if gws[2] == 0 {
        wd = 2;
    } else {
        wd = 3;
    }

    for (idx, arg) in args.into_iter().enumerate() {
        set_kernel_arg_both(&kernel, idx, arg.as_cvoid_ptr(), arg.size(), arg.is_num()).unwrap();    
    }
    enqueue_nd_range_kernel(&device.queue(), &kernel, wd, &gws, lws.as_ref(), None)?;
    Ok(())
}

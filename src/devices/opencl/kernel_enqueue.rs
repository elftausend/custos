use min_cl::api::{enqueue_nd_range_kernel, set_kernel_arg, OCLErrorKind};
use crate::{number::Number, Buffer, OpenCL};
use std::{ffi::c_void, mem::size_of};

pub trait AsClCvoidPtr {
    fn as_cvoid_ptr(&self) -> *const c_void;
    fn is_num(&self) -> bool {
        false
    }
    fn ptr_size(&self) -> usize {
        std::mem::size_of::<*const c_void>()
    }
}

impl<'a, T, const N: usize> AsClCvoidPtr for &Buffer<'a, T, OpenCL, N>
{
    fn as_cvoid_ptr(&self) -> *const c_void {
        self.ptr.ptr
    }
}

impl<'a, T, const N: usize> AsClCvoidPtr for Buffer<'a, T, OpenCL, N>
{
    fn as_cvoid_ptr(&self) -> *const c_void {
        self.ptr.ptr
    }
}

impl<T: Number> AsClCvoidPtr for T {
    fn as_cvoid_ptr(&self) -> *const c_void {
        self as *const T as *const c_void
    }

    fn ptr_size(&self) -> usize {
        size_of::<T>()
    }

    fn is_num(&self) -> bool {
        true
    }
}

pub fn enqueue_kernel(
    device: &OpenCL,
    src: &str,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    args: &[&dyn AsClCvoidPtr],
) -> crate::Result<()> {
    let kernel = device.kernel_cache.borrow_mut().kernel_cache(device, src)?;

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

    for (idx, arg) in args.iter().enumerate() {
        set_kernel_arg(
            &kernel,
            idx,
            arg.as_cvoid_ptr(),
            arg.ptr_size(),
            arg.is_num(),
        )
        .unwrap();
    }
    enqueue_nd_range_kernel(&device.queue(), &kernel, wd, &gws, lws.as_ref(), None)?;
    Ok(())
}

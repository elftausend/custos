use crate::{number::Number, Buffer, OnDropBuffer, OpenCL, Shape};
use min_cl::api::{enqueue_nd_range_kernel, set_kernel_arg, OCLErrorKind};
use std::{ffi::c_void, mem::size_of};

use super::CLPtr;

/// Converts `Self` to a *const c_void.
/// This enables taking `Buffer` and a number `T` as an argument to an OpenCL kernel.
/// # Example
/// ```
/// use custos::{OpenCL, Buffer, opencl::AsClCvoidPtr, Base};
///
/// fn args(args: &[&dyn AsClCvoidPtr]) {
///     // ...
/// }
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::<Base>::new(0)?;
///
///     let buf = Buffer::<f32, _>::new(&device, 10);
///     let num = 4;
///     args(&[&num, &buf]);
///     Ok(())
/// }
/// ```
pub trait AsClCvoidPtr {
    /// Converts `Self` to a *const c_void.
    /// # Example
    /// ```
    /// use custos::{OpenCL, Buffer, opencl::AsClCvoidPtr, Base};
    ///
    /// fn main() -> custos::Result<()> {
    ///     let device = OpenCL::<Base>::new(0)?;
    ///     let buf = Buffer::<f32, _>::new(&device, 10);
    ///     
    ///     let ptr = buf.as_cvoid_ptr();
    ///     assert_eq!(ptr, buf.cl_ptr());
    ///     Ok(())
    /// }
    ///
    fn as_cvoid_ptr(&self) -> *const c_void;

    /// Checks if `Self` is a number.
    /// # Example
    /// ```
    /// use custos::opencl::AsClCvoidPtr;
    ///
    /// assert_eq!(4f32.is_num(), true);
    /// ```
    #[inline]
    fn is_num(&self) -> bool {
        false
    }

    /// Returns the size of `Self` in bytes.
    /// # Example
    /// ```
    /// use custos::{OpenCL, Buffer, opencl::AsClCvoidPtr, Base};
    ///
    /// fn main() -> custos::Result<()> {
    ///     assert_eq!(4f32.ptr_size(), 4);    
    ///
    ///     let device = OpenCL::<Base>::new(0)?;
    ///
    ///     let buf = Buffer::<f32, _>::new(&device, 10);
    ///     assert_eq!(buf.ptr_size(), 8);
    ///     Ok(())
    /// }
    ///
    /// ```
    #[inline]
    fn ptr_size(&self) -> usize {
        std::mem::size_of::<*const c_void>()
    }
}

impl<'a, Mods: OnDropBuffer, T, S: Shape> AsClCvoidPtr for &Buffer<'a, T, OpenCL<Mods>, S> {
    #[inline]
    fn as_cvoid_ptr(&self) -> *const c_void {
        self.data.ptr
    }
}

impl<'a, Mods: OnDropBuffer, T, S: Shape> AsClCvoidPtr for Buffer<'a, T, OpenCL<Mods>, S> {
    #[inline]
    fn as_cvoid_ptr(&self) -> *const c_void {
        self.data.ptr
    }
}

impl<T: Number> AsClCvoidPtr for T {
    #[inline]
    fn as_cvoid_ptr(&self) -> *const c_void {
        self as *const T as *const c_void
    }

    #[inline]
    fn ptr_size(&self) -> usize {
        size_of::<T>()
    }

    #[inline]
    fn is_num(&self) -> bool {
        true
    }
}

impl<T> AsClCvoidPtr for CLPtr<T> {
    #[inline]
    fn as_cvoid_ptr(&self) -> *const c_void {
        self.ptr
    }
}

/// Executes a cached OpenCL kernel.
/// # Example
///
/// ```
/// use custos::{OpenCL, Buffer, opencl::enqueue_kernel, Base};
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::<Base>::new(0)?;
///     let mut buf = Buffer::<f32, _>::new(&device, 10);
///
///     enqueue_kernel(&device, "
///      __kernel void add(__global float* buf, float num) {
///         int idx = get_global_id(0);
///         buf[idx] += num;
///      }
///     ", [buf.len(), 0, 0], None, &[&mut buf, &4f32])?;
///     
///     assert_eq!(buf.read_to_vec(), [4.0; 10]);    
///
///     Ok(())
/// }
/// ```
pub fn enqueue_kernel<Mods>(
    device: &OpenCL<Mods>,
    src: &str,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    args: &[&dyn AsClCvoidPtr],
) -> crate::Result<()> {
    let mut binding = device.kernel_cache.borrow_mut();
    let kernel = binding.kernel(device, src)?;

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
            kernel,
            idx,
            arg.as_cvoid_ptr(),
            arg.ptr_size(),
            arg.is_num(),
        )?;
    }

    // with waitlist:
    // device.inner.enqueue_nd_range_kernel(kernel, wd, &gws, lws.as_ref(), None);
    enqueue_nd_range_kernel(device.queue(), kernel, wd, &gws, lws.as_ref(), None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // use core::ffi::c_void;

    use crate::{opencl::chosen_cl_idx, Base, Buffer, CDatatype, OpenCL};

    #[test]
    fn test_kernel_launch() -> crate::Result<()> {
        let device = OpenCL::<Base>::new(chosen_cl_idx())?;

        let src = format!("
            __kernel void add(__global const {datatype}* lhs, __global const {datatype}* rhs, __global {datatype}* out) 
            {{
                size_t id = get_global_id(0);
                out[id] = lhs[id] + rhs[id];
            }}
        ", datatype=f32::C_DTYPE_STR);

        let lhs = Buffer::from((&device, [1f32, 5.1, 1.2, 2.3, 4.6]));
        let rhs = Buffer::from((&device, [1f32, 5.1, 1.2, 2.3, 4.6]));

        let mut out = Buffer::<f32, _>::new(&device, lhs.len());

        device.launch_kernel(&src, [lhs.len(), 0, 0], None, &[&lhs, &rhs, &mut out])?;

        Ok(())
    }

    fn ew_add_kernel<T: CDatatype>(op: &str) -> String {
        format!(
            "__kernel void cl_ew(__global {datatype}* lhs, __global {datatype}* rhs, __global {datatype}* out) {{
                size_t idx = get_global_id(0);
    
                out[idx] = lhs[idx] {op} rhs[idx];
            }}"
        , datatype=T::C_DTYPE_STR)
    }

    #[test]
    fn test_get_work_group_size() -> crate::Result<()> {
        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        let mut kernel_cache = device.kernel_cache.borrow_mut();

        let kernel = kernel_cache.kernel(&device, &ew_add_kernel::<f32>("+"))?;

        let mut local = 0u64;

        unsafe {
            min_cl::api::ffi::clGetKernelWorkGroupInfo(
                kernel.0,
                device.inner.device.0,
                min_cl::api::ffi::CL_KERNEL_WORK_GROUP_SIZE,
                core::mem::size_of_val(&local),
                &mut local as *mut _ as *mut core::ffi::c_void,
                core::ptr::null_mut()
            );
        }

        println!("local: {local}");


        Ok(())
    }
    
}

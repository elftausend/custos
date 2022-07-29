use std::{ffi::c_void, ptr::null_mut};

pub use cl_cache::*;
pub use cl_device::*;
pub use cl_devices::*;
pub use kernel_options::*;

pub mod api;
mod cl_cache;
pub mod cl_device;
pub mod cl_devices;
mod kernel_options;

use self::api::{create_buffer, MemFlags};
use crate::{BufFlag, Buffer, CDatatype, Node, DeviceError};

/// Returns an OpenCL pointer that is bound to the host pointer stored in the specified buffer.
pub fn to_unified<T>(device: &CLDevice, no_drop: Buffer<T>) -> crate::Result<(*mut c_void, *const bool)> {
    // use the host pointer to create an OpenCL buffer
    let cl_ptr = create_buffer(
        &device.ctx(),
        MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
        no_drop.len,
        Some(&no_drop),
    )?;

    let valid = Box::leak(Box::new(true));

    let old_ptr = CL_CACHE.with(|cache| {
        // add created buffer to the "caching chain"
        cache.borrow_mut().nodes.insert(
            Node::new(no_drop.len),
            RawCL {
                ptr: cl_ptr,
                host_ptr: null_mut(),
                valid
            },
        )
    });

    // this pointer was overwritten previously, hence it can be deallocated
    // this line can be removed, however it shows that deallocating the old pointer makes sense
    drop(old_ptr);

    Ok((cl_ptr, valid))
}

/// Converts an 'only' CPU buffer into an OpenCL + CPU (unified memory) buffer.
pub fn construct_buffer<T>(
    device: &CLDevice,
    no_drop: Buffer<T>,
) -> crate::Result<Buffer<T>> {

    if no_drop.flag == BufFlag::None || no_drop.flag == BufFlag::Wrapper {
        return Err(DeviceError::ConstructError.into())
    }

    let (host_ptr, len) = (no_drop.host_ptr(), no_drop.len);
    let (cl_ptr, valid) = to_unified(device, no_drop)?;
    
    Ok(Buffer {
        ptr: (host_ptr, cl_ptr, 0),
        len,
        flag: BufFlag::Cache(valid),
    })
}

/// Sets the elements of an OpenCL Buffer to zero.
/// # Example
/// ```
/// use custos::{CLDevice, Buffer, VecRead, opencl::cl_clear};
///
/// fn main() -> Result<(), custos::Error> {
///     let device = CLDevice::new(0)?;
///     let mut lhs = Buffer::<i16>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(device.read(&lhs), vec![15, 30, 21, 5, 8]);
///
///     cl_clear(&device, &mut lhs);
///     assert_eq!(device.read(&lhs), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn cl_clear<T: CDatatype>(device: &CLDevice, lhs: &mut Buffer<T>) -> crate::Result<()> {
    let src = format!(
        "
        __kernel void clear(__global {datatype}* self) {{
            size_t id = get_global_id(0);
            self[id] = 0;
        }}
    ",
        datatype = T::as_c_type_str()
    );

    let gws = [lhs.len, 0, 0];
    enqueue_kernel(&device, &src, gws, None, &[lhs])?;
    Ok(())
    //enqueue_kernel(device, &src, gws, None, vec![lhs])
}

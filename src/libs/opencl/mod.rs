use std::ffi::c_void;

pub use cl_cache::*;
pub use cl_device::*;
pub use cl_devices::*;
pub use kernel_options::*;

pub mod api;
// TODO: remove ops (mind clear op)
mod cl_cache;
pub mod cl_device;
pub mod cl_devices;
mod kernel_options;

use self::api::{create_buffer, release_mem_object, MemFlags};
use crate::{Buffer, Node, CDatatype, BufFlag};

/// Returns an OpenCL pointer that is bound to the host pointer stored in the specified matrix.
pub fn to_unified<T>(device: &CLDevice, no_drop: Buffer<T>) -> crate::Result<*mut c_void> {
    // use the host pointer to create an OpenCL buffer
    let cl_ptr = create_buffer(
        &device.ctx(),
        MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
        no_drop.len,
        Some(&no_drop),
    )?;

    let old_ptr = CL_CACHE.with(|cache| {
        // add created buffer to the "caching chain"
        cache
            .borrow_mut()
            .nodes
            .insert(Node::new(no_drop.len), RawCL { ptr: cl_ptr, len: no_drop.len })
    });

    // this pointer was overwritten previously, hence it can be deallocated
    if let Some(old) = old_ptr {
        unsafe {
            release_mem_object(old.ptr)?;
        }
    };

    Ok(cl_ptr)
}

/// Converts an 'only' CPU buffer into an OpenCL + CPU (unified memory) buffer.
pub fn construct_buffer<T>(
    device: &CLDevice,
    cpu: &crate::CPU,
    no_drop: Buffer<T>,
) -> crate::Result<Buffer<T>> {
    let (host_ptr, len) = (no_drop.host_ptr(), no_drop.len);
    let cl_ptr = to_unified(device, no_drop)?;
    // TODO: When should the buffer be freed, if the "safe" feature is used?

    // Both lines prevent the deallocation of the underlying buffer.
    //Box::into_raw(Box::new(no_drop)); // "safe" mode
    // TODO: Deallocate cpu buffer? This may leak memory.
    cpu.inner.borrow_mut().ptrs.clear(); // default mode

    Ok(Buffer {
        ptr: (host_ptr, cl_ptr, 0),
        len,
        flag: BufFlag::Cache,
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
    KernelOptions::<T>::new(device, lhs, gws, &src)?.run()?;
    Ok(())
    //enqueue_kernel(device, &src, gws, None, vec![lhs])
}

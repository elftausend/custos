use std::ffi::c_void;

pub use cl_cache::*;
pub use cl_device::*;
pub use cl_devices::*;
pub use kernel_options::*;
pub use ops::*;

pub mod api;
// TODO: remove ops (mind clear op)
pub mod ops;
pub mod cl_device;
pub mod cl_devices;
mod kernel_options;
mod cl_cache;

use crate::{Node, Buffer};
use self::api::{create_buffer, MemFlags, release_mem_object};

/// Returns an OpenCL pointer that is bound to the host pointer stored in the specified matrix.
pub fn to_unified<T>(device: &CLDevice, no_drop: Buffer<T>) -> crate::Result<*mut c_void> {
    // use the host pointer to create an OpenCL buffer
    let cl_ptr = create_buffer(
        &device.ctx(), 
        MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
        no_drop.size(), 
        Some(&no_drop)
    )?;

    let old_ptr = CL_CACHE.with(|cache| {
        // add created buffer to the "caching chain"
        cache.borrow_mut().nodes.insert(Node::new(no_drop.size()), (OclPtr(cl_ptr), no_drop.size()))
    });

    // this pointer was overwritten previously, hence can it be deallocated
    if let Some(old) = old_ptr {
        unsafe {
            release_mem_object(old.0.0)?;
        }
    };

    Ok(cl_ptr)
}

/// Converts an 'only' CPU buffer into an OpenCL + CPU (unified memory) buffer.
pub fn construct_buffer<T>(device: &CLDevice, cpu: &crate::CPU, no_drop: Buffer<T>) -> crate::Result<Buffer<T>>  {
    let (host_ptr, len) = (no_drop.host_ptr(), no_drop.len);
    let cl_ptr = to_unified(device, no_drop)?;
    // TODO: When should the buffer be freed, if the "safe" feature is used?

    // Both lines prevent the deallocation of the underlying buffer.
    //Box::into_raw(Box::new(no_drop)); // "safe" mode
    // TODO: Deallocate cpu buffer? This may leak memory.
    cpu.inner.borrow_mut().ptrs.clear(); // default mode
    
    Ok(Buffer {
        ptr: (host_ptr, cl_ptr, 0),
        len
    })
}
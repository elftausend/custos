use std::{ffi::c_void, rc::Rc};

pub use cl_device::*;
pub use cl_devices::*;
pub use kernel_cache::*;
pub use kernel_enqueue::*;

pub mod api;
pub mod cl_device;
pub mod cl_devices;
mod kernel_cache;
mod kernel_enqueue;

#[cfg(not(feature = "realloc"))]
use crate::{AddGraph, BufFlag, DeviceError, GraphReturn};

#[cfg(not(feature = "realloc"))]
use std::fmt::Debug;

use self::api::{create_buffer, MemFlags};
use crate::{Buffer, CDatatype, Ident, Node, CPU};

/// Returns an OpenCL pointer that is bound to the host pointer stored in the specified buffer.
pub fn to_unified<T>(
    device: &OpenCL,
    no_drop: Buffer<T, CPU>,
    graph_node: Node,
) -> crate::Result<*mut c_void> {
    // use the host pointer to create an OpenCL buffer
    let cl_ptr = create_buffer(
        &device.ctx(),
        MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
        no_drop.len,
        Some(&no_drop),
    )?;

    let old_ptr = device.cache.borrow_mut().nodes.insert(
        Ident::new(no_drop.len),
        Rc::new(RawCL {
            ptr: cl_ptr,
            host_ptr: no_drop.host_ptr() as *mut u8,
            node: graph_node,
        }),
    );

    // this pointer was overwritten previously, hence can it be deallocated
    // this line can be removed, however it shows that deallocating the old pointer makes sense
    drop(old_ptr);

    Ok(cl_ptr)
}

#[cfg(not(feature = "realloc"))]
/// Converts an 'only' CPU buffer into an OpenCL + CPU (unified memory) buffer.
/// # Safety
/// The pointer of the no_drop Buffer must be valid for the entire lifetime of the returned Buffer.
pub unsafe fn construct_buffer<'a, T: Debug>(
    device: &'a OpenCL,
    no_drop: Buffer<T, OpenCL>,
    add_node: impl AddGraph,
) -> crate::Result<Buffer<'a, T, OpenCL>> {
    use crate::bump_count;

    if no_drop.flag == BufFlag::None {
        return Err(DeviceError::ConstructError.into());
    }

    if let Some(rawcl) = device.cache.borrow().nodes.get(&Ident::new(no_drop.len)) {
        return Ok(Buffer {
            ptr: (rawcl.host_ptr as *mut T, rawcl.ptr, 0),
            len: no_drop.len,
            device: Some(device),
            flag: BufFlag::Cache,
            node: rawcl.node,
        });
    }

    let graph_node = device.graph().add(no_drop.len, add_node);

    let (host_ptr, len) = (no_drop.host_ptr(), no_drop.len);
    let cl_ptr = to_unified(device, no_drop, graph_node)?;

    bump_count();

    Ok(Buffer {
        ptr: (host_ptr, cl_ptr, 0),
        len,
        device: device.dev(),
        flag: BufFlag::Cache,
        node: graph_node,
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
pub fn cl_clear<T: CDatatype>(device: &OpenCL, lhs: &mut Buffer<T, OpenCL>) -> crate::Result<()> {
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
    enqueue_kernel(device, &src, gws, None, &[lhs])?;
    Ok(())
}

use std::{ffi::c_void, rc::Rc};

#[cfg(not(feature = "realloc"))]
use crate::{AddGraph, BufFlag, DeviceError, GraphReturn};

#[cfg(not(feature = "realloc"))]
use std::fmt::Debug;

use super::{
    api::{create_buffer, MemFlags},
    RawCL,
};
use crate::{Buffer, Ident, Node, OpenCL, CPU};

/// Returns an OpenCL pointer that is bound to the host pointer stored in the specified buffer.
/// This function is used in the `constuct_buffer()` function.
/// # Safety
/// The host pointer inside the no_drop `Buffer` must live as long as the resulting pointer.
pub unsafe fn to_unified<T>(
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
/// The host pointer of the no_drop `Buffer` must be valid for the entire lifetime of the returned Buffer.
///
/// # Example
/// ```
/// use custos::prelude::*;
///
/// fn main() -> custos::Result<()> {
///     let cpu = CPU::new();
///     let mut no_drop: Buffer = cpu.cached(4);
///     no_drop.write(&[1., 3.1, 2.34, 0.76]);
///     
///     let device = OpenCL::new(0)?;
///     let buf = unsafe {
///         construct_buffer(&device, no_drop, ())?
///     };
///     
///     assert_eq!(buf.read(), vec![1., 3.1, 2.34, 0.76]);
///     assert_eq!(buf.as_slice(), &[1., 3.1, 2.34, 0.76]);
///     Ok(())
/// }
/// ```
pub unsafe fn construct_buffer<'a, T: Debug>(
    device: &'a OpenCL,
    no_drop: Buffer<T, CPU>,
    add_node: impl AddGraph,
) -> crate::Result<Buffer<'a, T, OpenCL>> {
    use crate::{bump_count, opencl::CLPtr, PtrType};

    if no_drop.flag == BufFlag::None {
        return Err(DeviceError::ConstructError.into());
    }

    // if buffer was already converted, return the cache entry.
    if let Some(rawcl) = device.cache.borrow().nodes.get(&Ident::new(no_drop.len)) {
        return Ok(Buffer {
            ptr: CLPtr {
                ptr: rawcl.ptr,
                host_ptr: rawcl.host_ptr as *mut T,
            },
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
        ptr: CLPtr::from_ptrs((host_ptr, cl_ptr, 0)),
        len,
        device: Some(device),
        flag: BufFlag::Cache,
        node: graph_node,
    })
}

#[cfg(test)]
mod tests {
    use crate::{opencl::CLPtr, BufFlag, Buffer, CacheBuf, Node, OpenCL, PtrType, CPU};

    use super::{construct_buffer, to_unified};

    #[test]
    fn test_to_unified() -> crate::Result<()> {
        let cpu = CPU::new();
        let mut no_drop: Buffer = cpu.cached(3);
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::new(0)?;

        let (host_ptr, len) = (no_drop.host_ptr(), no_drop.len);
        let cl_host_ptr = unsafe { to_unified(&device, no_drop, Node::default())? };

        let buf: Buffer<f32, OpenCL> = Buffer {
            ptr: CLPtr::from_ptrs((host_ptr, cl_host_ptr, 0)),
            len,
            device: Some(&device),
            flag: BufFlag::Cache,
            node: Node::default(),
        };

        assert_eq!(buf.read(), vec![1., 2.3, 0.76]);
        assert_eq!(buf.as_slice(), &[1., 2.3, 0.76]);
        Ok(())
    }

    #[test]
    fn test_construct_buffer() -> crate::Result<()> {
        let cpu = CPU::new();
        let mut no_drop: Buffer = cpu.cached(3);
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::new(0)?;
        let buf = unsafe { construct_buffer(&device, no_drop, ())? };

        assert_eq!(buf.read(), vec![1., 2.3, 0.76]);
        assert_eq!(buf.as_slice(), &[1., 2.3, 0.76]);

        Ok(())
    }
}

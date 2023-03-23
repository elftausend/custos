use std::{ffi::c_void, rc::Rc};

#[cfg(not(feature = "realloc"))]
use crate::{AddGraph, AllocFlag, DeviceError, GraphReturn};

use super::CLPtr;
use crate::{Buffer, Ident, OpenCL, Shape, CPU};
use min_cl::api::{create_buffer, MemFlags};

/// Returns an OpenCL pointer that is bound to the host pointer stored in the specified buffer.
/// This function is used in the `constuct_buffer()` function.
/// # Safety
/// The host pointer inside the no_drop `Buffer` must live as long as the resulting pointer.
pub unsafe fn to_cached_unified<T, S: Shape>(
    device: &OpenCL,
    no_drop: Buffer<T, CPU, S>,
) -> crate::Result<*mut c_void> {
    // use the host pointer to create an OpenCL buffer
    let cl_ptr = create_buffer(
        device.ctx(),
        MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
        no_drop.len(),
        Some(&no_drop),
    )?;

    let old_ptr = device.addons.cache.borrow_mut().nodes.insert(
        Ident::new(no_drop.len()),
        Rc::new(CLPtr {
            ptr: cl_ptr,
            host_ptr: no_drop.host_ptr() as *mut u8,
            len: no_drop.len(),
            flag: AllocFlag::None,
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
#[cfg_attr(unified_cl, doc = "```")]
#[cfg_attr(not(unified_cl), doc = "```ignore")]
/// use custos::prelude::*;
///
/// fn main() -> custos::Result<()> {
///     let cpu = CPU::new();
///     let mut no_drop: Buffer = cpu.retrieve(4, ());
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
pub unsafe fn construct_buffer<'a, T, S: Shape>(
    device: &'a OpenCL,
    mut no_drop: Buffer<T, CPU, S>,
    add_node: impl AddGraph,
) -> crate::Result<Buffer<'a, T, OpenCL, S>> {
    use crate::bump_count;

    if no_drop.ptr.flag == AllocFlag::None {
        return Err(DeviceError::ConstructError.into());
    }

    // if buffer was already converted, return the cache entry.
    if let Some(rawcl) = device
        .addons
        .cache
        .borrow()
        .nodes
        .get(&Ident::new(no_drop.len()))
    {
        return Ok(Buffer {
            ptr: CLPtr {
                ptr: rawcl.ptr,
                host_ptr: rawcl.host_ptr as *mut T,
                len: no_drop.len(),
                flag: no_drop.ptr.flag,
            },
            device: Some(device),
            ident: Ident::new(no_drop.len()),
            requires_grad: false,
        });
    }

    // TODO: remove
    let graph_node = device.graph_mut().add(no_drop.len(), add_node);

    let (host_ptr, len) = (no_drop.host_ptr_mut(), no_drop.len());
    let ptr = to_cached_unified(device, no_drop)?;

    bump_count();

    Ok(Buffer {
        ptr: CLPtr {
            ptr,
            host_ptr,
            len,
            flag: AllocFlag::Wrapper,
        },
        device: Some(device),
        ident: Ident {
            idx: *device.graph_mut().idx_trans.get(&graph_node.idx).unwrap(),
            len,
        },
        requires_grad: false,
    })
}

#[cfg(unified_cl)]
#[cfg(test)]
mod tests {
    use crate::{opencl::CLPtr, AllocFlag, Buffer, Device, Ident, OpenCL, CPU};

    use super::{construct_buffer, to_cached_unified};

    #[test]
    fn test_to_unified() -> crate::Result<()> {
        let cpu = CPU::new();
        let mut no_drop: Buffer = cpu.retrieve(3, ());
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::new(0)?;

        let (host_ptr, len) = (no_drop.host_ptr_mut(), no_drop.len());
        let cl_host_ptr = unsafe { to_cached_unified(&device, no_drop)? };

        let buf: Buffer<f32, OpenCL> = Buffer {
            ptr: CLPtr {
                ptr: cl_host_ptr,
                host_ptr,
                len,
                flag: AllocFlag::Wrapper,
            },
            device: Some(&device),
            ident: Ident::new_bumped(len),
            requires_grad: false,
        };

        assert_eq!(buf.read(), vec![1., 2.3, 0.76]);
        assert_eq!(buf.as_slice(), &[1., 2.3, 0.76]);
        Ok(())
    }

    #[test]
    fn test_construct_buffer() -> crate::Result<()> {
        let cpu = CPU::new();
        let mut no_drop: Buffer = cpu.retrieve(3, ());
        no_drop.write(&[1., 2.3, 0.76]);

        let device = OpenCL::new(0)?;
        let buf = unsafe { construct_buffer(&device, no_drop, ())? };

        assert_eq!(buf.read(), vec![1., 2.3, 0.76]);
        assert_eq!(buf.as_slice(), &[1., 2.3, 0.76]);

        Ok(())
    }

    #[cfg(unified_cl)]
    #[cfg(not(feature = "realloc"))]
    #[test]
    fn test_cpu_to_unified_leak() -> crate::Result<()> {
        use std::{collections::HashMap, hash::BuildHasherDefault, rc::Rc};

        use crate::{range, set_count, Device, Ident, IdentHasher};

        let cl_dev = OpenCL::new(0)?;

        unsafe { set_count(0) };

        for _ in range(10) {
            let cl_cpu_buf = {
                let cpu = CPU::new();
                let mut buf = cpu.retrieve::<i32, ()>(6, ());
                buf.copy_from_slice(&[1, 2, 3, 4, 5, 6]);

                let cl_cpu_buf = unsafe { crate::opencl::construct_buffer(&cl_dev, buf, ())? };
                let mut hm = HashMap::<Ident, _, BuildHasherDefault<IdentHasher>>::default();
                std::mem::swap(&mut cpu.addons.cache.borrow_mut().nodes, &mut hm);
                for mut value in hm {
                    let ptr = Rc::get_mut(&mut value.1).unwrap();
                    ptr.ptr = std::ptr::null_mut();
                }
                cl_cpu_buf
            };
            assert_eq!(cl_cpu_buf.as_slice(), &[1, 2, 3, 4, 5, 6]);
            assert_eq!(cl_cpu_buf.read(), &[1, 2, 3, 4, 5, 6]);
        }
        Ok(())
    }
}

use super::{cpu_exec_binary, cpu_exec_unary};
use crate::{Buffer, OpenCL, CPU};

#[cfg(not(feature = "realloc"))]
use crate::opencl::construct_buffer;

pub fn cpu_exec_unary_may_unified<'a, T, F>(
    device: &'a OpenCL,
    x: &Buffer<T, OpenCL>,
    f: F,
) -> crate::Result<Buffer<'a, T, OpenCL>>
where
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &Buffer<'_, T, CPU>) -> Buffer<'b, T, CPU>,
{
    // TODO: use compile time unified_cl flag -> get from custos?
    #[cfg(not(feature = "realloc"))]
    if device.unified_mem() {
        // Using a CPU stored in a OpenCL in order to get a (correct) cache entry.
        // Due to the (new) caching architecture, using a new CPU isn't possible,
        // as the cache would be newly created every iteration.

        // host ptr buffer
        let no_drop = f(&device.cpu, &unsafe {
            Buffer::from_raw_host(x.ptr.host_ptr, x.len())
        });

        // convert host ptr / CPU buffer into a host ptr + OpenCL ptr buffer
        return unsafe {
            construct_buffer(device, no_drop, /*buf.node.idx*/ ())
        };
    }

    #[cfg(feature = "realloc")]
    if device.unified_mem() {
        let cpu = CPU::new();
        return Ok(Buffer::from((
            device,
            f(&cpu, &unsafe {Buffer::from_raw_host(x.ptr.host_ptr, x.len())}),
        )));
    }

    cpu_exec_unary(device, x, f)
}

pub fn cpu_exec_binary_may_unified<'a, T, F>(
    device: &'a OpenCL,
    lhs: &Buffer<T, OpenCL>,
    rhs: &Buffer<T, OpenCL>,
    f: F,
) -> crate::Result<Buffer<'a, T, OpenCL>>
where
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>) -> Buffer<'b, T, CPU>,
{
    // TODO: use compile time unified_cl flag -> get from custos?
    #[cfg(not(feature = "realloc"))]
    if device.unified_mem() {
        // Using a CPU stored in a OpenCL in order to get a (correct) cache entry.
        // Due to the (new) caching architecture, using a new CPU isn't possible,
        // as the cache would be newly created every iteration.

        // host ptr buffer
        let no_drop = f(
            &device.cpu,
            &unsafe { Buffer::from_raw_host(lhs.ptr.host_ptr, lhs.len()) },
            &unsafe { Buffer::from_raw_host(rhs.ptr.host_ptr, rhs.len()) },
        );

        // convert host ptr / CPU buffer into a host ptr + OpenCL ptr buffer
        return unsafe {
            construct_buffer(device, no_drop, /*buf.node.idx*/ ())
        };
    }

    #[cfg(feature = "realloc")]
    if device.unified_mem() {
        let cpu = CPU::new();
        return Ok(Buffer::from((
            device,
            f(
                &cpu,
                &unsafe { Buffer::from_raw_host(lhs.ptr.host_ptr, lhs.len()) },
                &unsafe { Buffer::from_raw_host(rhs.ptr.host_ptr, rhs.len()) },
            ),
        )));
    }

    cpu_exec_binary(device, lhs, rhs, f)
}

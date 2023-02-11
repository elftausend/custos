use crate::{CPU, Buffer, OpenCL, opencl::construct_buffer};

use super::cpu_exec_unary;


pub fn cpu_exec_unary_may_unified<'a, T, F>(device: &'a OpenCL, x: &Buffer<T, OpenCL>, f: F) -> crate::Result<Buffer<'a, T, OpenCL>>
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
        let no_drop = f(
            &device.cpu,
            &unsafe {Buffer::from_raw_host(x.ptr.host_ptr, x.len())},
        );

        // convert host ptr / CPU buffer into a host ptr + OpenCL ptr buffer
        return unsafe {
            construct_buffer(device, no_drop, /*buf.node.idx*/ ())
        };
    }

    #[cfg(feature = "realloc")]
    if device.unified_mem() {
        let cpu = CPU::new();
        return Ok(Matrix::from((
            device,
            f(&cpu, &Matrix::from((matrix.ptr.host_ptr, matrix.dims))),
        )));
    }

    cpu_exec_unary(device, x, f)
}

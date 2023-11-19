use super::{
    cpu_exec_binary, cpu_exec_binary_mut, cpu_exec_reduce, cpu_exec_unary, cpu_exec_unary_mut,
};
use crate::{Buffer, CachedCPU, OnDropBuffer, OpenCL, Retrieve, UnifiedMemChain, CPU};

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec_unary], as new memory is not allocated.
///
/// `cpu_exec_unary_may_unified` can be used interchangeably with [cpu_exec_unary].
#[track_caller]
pub fn cpu_exec_unary_may_unified<
    'a,
    T,
    F,
    Mods: OnDropBuffer + Retrieve<OpenCL<Mods>, T> + UnifiedMemChain<OpenCL<Mods>> + 'static,
>(
    device: &'a OpenCL<Mods>,
    x: &Buffer<T, OpenCL<Mods>>,
    f: F,
) -> crate::Result<Buffer<'a, T, OpenCL<Mods>>>
where
    T: Clone + Default + 'static,
    F: for<'b> Fn(&'b CachedCPU, &Buffer<'_, T, CachedCPU>) -> Buffer<'b, T, CachedCPU>,
{
    // TODO: use compile time unified_cl flag -> get from custos?
    #[cfg(not(feature = "realloc"))]
    if device.unified_mem() {
        // Using a CPU stored in a OpenCL in order to get a (correct) cache entry.
        // Due to the (new) caching architecture, using a new CPU isn't possible,
        // as the cache would be newly created every iteration.

        // host ptr buffer
        let no_drop = f(&device.cpu, &unsafe {
            Buffer::from_raw_host(x.data.host_ptr, x.len())
        });

        // convert host ptr / CPU buffer into a host ptr + OpenCL ptr buffer
        return device.construct_unified_buf_from_cpu_buf(device, no_drop);
        /*return unsafe {
            construct_buffer(device, no_drop, /*buf.node.idx*/ ())
        };*/
    }

    #[cfg(feature = "realloc")]
    if device.unified_mem() {
        let cpu = CPU::<Base>::new();
        return Ok(Buffer::from((
            device,
            f(&cpu, &unsafe {
                Buffer::from_raw_host(x.ptr.host_ptr, x.len())
            }),
        )));
    }
    // TODO: add to graph?:     convert.node = device.graph().add(convert.len(), matrix.node.idx);
    cpu_exec_unary(device, x, f)
}

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec_unary_mut], as new memory is not allocated.
///
/// `cpu_exec_unary_may_unified` can be used interchangeably with [cpu_exec_unary_mut].
pub fn cpu_exec_unary_may_unified_mut<'a, T, F, Mods: OnDropBuffer + 'static>(
    device: &'a OpenCL<Mods>,
    lhs: &mut Buffer<T, OpenCL<Mods>>,
    f: F,
) -> crate::Result<()>
where
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>),
{
    let cpu = CPU::<crate::Base>::new();

    if device.unified_mem() {
        return {
            f(&cpu, &mut unsafe {
                Buffer::from_raw_host(lhs.data.host_ptr, lhs.len())
            });
            Ok(())
        };
    }

    // TODO: add to graph?:     convert.node = device.graph().add(convert.len(), matrix.node.idx);
    cpu_exec_unary_mut(device, lhs, f)
}

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec_binary], as new memory is not allocated.
///
/// `cpu_exec_binary_may_unified` can be used interchangeably with [cpu_exec_binary].
pub fn cpu_exec_binary_may_unified<
    'a,
    T,
    F,
    Mods: OnDropBuffer + UnifiedMemChain<OpenCL<Mods>> + Retrieve<OpenCL<Mods>, T> + 'static,
>(
    device: &'a OpenCL<Mods>,
    lhs: &Buffer<T, OpenCL<Mods>>,
    rhs: &Buffer<T, OpenCL<Mods>>,
    f: F,
) -> crate::Result<Buffer<'a, T, OpenCL<Mods>>>
where
    T: Clone + Default + 'static,
    F: for<'b> Fn(
        &'b CachedCPU,
        &Buffer<'_, T, CachedCPU>,
        &Buffer<'_, T, CachedCPU>,
    ) -> Buffer<'b, T, CachedCPU>,
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
            &unsafe { Buffer::from_raw_host(lhs.data.host_ptr, lhs.len()) },
            &unsafe { Buffer::from_raw_host(rhs.data.host_ptr, rhs.len()) },
        );

        // convert host ptr / CPU buffer into a host ptr + OpenCL ptr buffer
        return device.construct_unified_buf_from_cpu_buf(device, no_drop);
        /*return unsafe {
            construct_buffer(device, no_drop, /*buf.node.idx*/ ())
        };*/
    }

    #[cfg(feature = "realloc")]
    if device.unified_mem() {
        let cpu = CPU::<Base>::new();
        return Ok(Buffer::from((
            device,
            f(
                &cpu,
                &unsafe { Buffer::from_raw_host(lhs.data.host_ptr, lhs.len()) },
                &unsafe { Buffer::from_raw_host(rhs.data.host_ptr, rhs.len()) },
            ),
        )));
    }

    Ok(cpu_exec_binary(device, lhs, rhs, f))
}

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec_binary_mut], as new memory is not allocated.
///
/// `cpu_exec_binary_may_unified` can be used interchangeably with [cpu_exec_binary_mut].
pub fn cpu_exec_binary_may_unified_mut<'a, T, F, Mods: OnDropBuffer + 'static>(
    device: &'a OpenCL<Mods>,
    lhs: &mut Buffer<T, OpenCL<Mods>>,
    rhs: &Buffer<T, OpenCL<Mods>>,
    f: F,
) -> crate::Result<()>
where
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>),
{
    let cpu = CPU::<crate::Base>::new();

    if device.unified_mem() {
        return {
            f(
                &cpu,
                &mut unsafe { Buffer::from_raw_host(lhs.data.host_ptr, lhs.len()) },
                &unsafe { Buffer::from_raw_host(rhs.data.host_ptr, rhs.len()) },
            );
            Ok(())
        };
    }

    cpu_exec_binary_mut(device, lhs, rhs, f)?;

    Ok(())
}

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec_reduce], as new memory is not allocated.
///
/// `cpu_exec_binary_may_unified` can be used interchangeably with [cpu_exec_reduce].
pub fn cpu_exec_reduce_may_unified<T, F>(device: &OpenCL, x: &Buffer<T, OpenCL>, f: F) -> T
where
    T: Default + Clone,
    F: Fn(&CPU, &Buffer<T, CPU>) -> T,
{
    let cpu = CPU::<crate::Base>::new();

    if device.unified_mem() {
        return f(&cpu, &unsafe {
            Buffer::from_raw_host(x.data.host_ptr, x.len())
        });
    }
    cpu_exec_reduce(x, f)
}

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec](crate::cpu_exec), as new memory is not allocated.
///
/// TODO
/// Syntax is different from [cpu_exec](crate::cpu_exec)!
#[macro_export]
macro_rules! cl_cpu_exec_unified {
    ($device:ident, $($t:ident),*; $op:expr) => {{
        // TODO add to graph?:     convert.node = device.graph().add(convert.len(), matrix.node.idx);
        let cpu = CPU::<Base>::new();
        if $device.unified_mem() {

            $crate::to_raw_host!($crate::CPU::<$crate::CachedModule<$crate::Base, $crate::CPU>>, $($t),*);

            #[cfg(not(feature = "realloc"))]
            {
                $device.construct_unified_buf_from_cpu_buf(&$device, $op)
                // unsafe {
                //     // TODO mind graph opt trace -> ()
                //     $crate::opencl::construct_buffer(&$device, $op, ())
                // }
            }

            #[cfg(feature = "realloc")]
            {
                let buf = Buffer::from((&$device, $op));
                $device.cpu.modules.cache.borrow_mut().nodes.clear();
                buf
            }

        } else {
            let buf = $crate::cpu_exec!($device, cpu, $($t),*; $op);
            $device.cpu.modules.cache.borrow_mut().nodes.clear();
            Ok(buf)
        }
    }};
}

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec_mut](crate::cpu_exec_mut), as new memory is not allocated.
///
/// TODO
/// Syntax is different from [cpu_exec](crate::cpu_exec)!
#[macro_export]
macro_rules! cl_cpu_exec_unified_mut {
    ($device:ident, $($t:ident),* WRITE_TO<$($write_to:ident, $from:ident),*> $op:expr) => {{
        // TODO: add to graph?:     convert.node = device.graph().add(convert.len(), matrix.node.idx);
        if $device.unified_mem() {
            $crate::to_raw_host!($crate::CPU::<$crate::CachedModule<$crate::Base, $crate::CPU>>, $($t),*);
            $crate::to_raw_host_mut!($crate::CPU::<$crate::CachedModule<$crate::Base, $crate::CPU>>, $($write_to, $from),*);
            $op;

        } else {
            let cpu = $crate::CPU::<$crate::Cached<Base>>::new();
            $crate::cpu_exec_mut!($device, cpu, $($t),* WRITE_TO<$($write_to, $from),*> $op);
            $device.cpu.modules.cache.borrow_mut().nodes.clear();
        }
    }};
}

#[cfg(test)]
mod tests {
    use crate::{Base, Device, OpenCL, WriteBuf};

    #[cfg(unified_cl)]
    #[test]
    fn test_cl_cpu_exec_unified_mut() {
        let device = OpenCL::<Base>::new(0).unwrap();
        let buf = device.buffer([1, 2, 3, 4, 5]);
        let mut out = device.buffer::<i32, (), _>(5);
        let out = &mut out;

        cl_cpu_exec_unified_mut!(device, buf WRITE_TO<out, out_cpu> {
            for (out, buf) in out_cpu.iter_mut().zip(buf.iter()) {
                *out += buf + 1;
            }
        });

        assert_eq!(out.read(), [2, 3, 4, 5, 6,]);
    }
}

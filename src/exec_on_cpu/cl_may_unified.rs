use super::{cpu_exec_binary_mut, cpu_exec_reduce, cpu_exec_unary_mut};
use crate::{Buffer, CachedCPU, OnDropBuffer, OpenCL, Retrieve, UnifiedMemChain, Unit, CPU};

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec_unary], as new memory is not allocated.
///
/// `cpu_exec_unary_may_unified` can be used interchangeably with [cpu_exec_unary].
pub fn cpu_exec_unary_may_unified<'a, T, F, Mods>(
    device: &'a OpenCL<Mods>,
    x: &Buffer<T, OpenCL<Mods>>,
    f: F,
) -> crate::Result<Buffer<'a, T, OpenCL<Mods>>>
where
    T: Unit + Clone + Default + 'static,
    F: for<'b> Fn(&'b CachedCPU, &Buffer<'_, T, CachedCPU>) -> Buffer<'b, T, CachedCPU>,
    Mods: OnDropBuffer + Retrieve<OpenCL<Mods>, T> + UnifiedMemChain<OpenCL<Mods>> + 'static,
{
    let cpu = &device.cpu;
    crate::cl_cpu_exec_unified!(device, cpu, x; f(&cpu, &x))
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
    T: Unit + Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>),
{
    let cpu = CPU::<crate::Base>::new();

    if device.unified_mem() {
        return {
            f(&cpu, &mut unsafe {
                Buffer::from_raw_host_device(&cpu, lhs.base().host_ptr, lhs.len())
            });
            Ok(())
        };
    }

    cpu_exec_unary_mut(device, lhs, f)
}

type CpuBuf<'a, T> = Buffer<'a, T, CachedCPU>;

/// If the current device supports unified memory, data is not deep-copied.
/// This is way faster than [cpu_exec_binary], as new memory is not allocated.
///
/// `cpu_exec_binary_may_unified` can be used interchangeably with [cpu_exec_binary].
pub fn cpu_exec_binary_may_unified<'a, T, F, Mods>(
    device: &'a OpenCL<Mods>,
    lhs: &Buffer<T, OpenCL<Mods>>,
    rhs: &Buffer<T, OpenCL<Mods>>,
    f: F,
) -> crate::Result<Buffer<'a, T, OpenCL<Mods>>>
where
    T: Unit + Clone + Default + 'static,
    F: for<'b> Fn(&'b CachedCPU, &CpuBuf<'_, T>, &CpuBuf<'_, T>) -> CpuBuf<'b, T>,
    Mods: UnifiedMemChain<OpenCL<Mods>> + Retrieve<OpenCL<Mods>, T> + 'static,
{
    let cpu = &device.cpu;
    crate::cl_cpu_exec_unified!(device, cpu, lhs, rhs; f(&cpu, &lhs, &rhs))
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
    T: Unit + Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>),
{
    let cpu = CPU::<crate::Base>::new();

    if device.unified_mem() {
        return {
            f(
                &cpu,
                &mut unsafe { Buffer::from_raw_host_device(&cpu, lhs.base().host_ptr, lhs.len()) },
                &unsafe { Buffer::from_raw_host_device(&cpu, rhs.base().host_ptr, rhs.len()) },
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
    T: Unit + Default + Clone,
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
#[macro_export]
macro_rules! cl_cpu_exec_unified {
    ($device:expr, $cpu:expr, $($t:ident),*; $op:expr) => {{
        if $device.unified_mem() {
            // Using a CPU stored in a OpenCL in order to get a (correct) cache entry.
            // Due to the (new) caching architecture, using a new CPU isn't possible,
            // as the cache would be newly created every iteration.

            $crate::to_raw_host!(&$device.cpu, $($t),*);
            $device.construct_unified_buf_from_cpu_buf(&$device, $op)

            // TODO if the cached module is not used, consider this:
            // {
            //     let buf = Buffer::from((&$device, $op));
            //     $device.cpu.modules.cache.borrow_mut().nodes.clear();
            //     buf
            // }

        } else {
            let buf = $crate::cpu_exec!($device, $cpu, $($t),*; $op);
            // would deallocate allocations, if retrieve on device.cpu was used in operation $op
            // $device.cpu.modules.cache.borrow_mut().nodes.clear();
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
    ($device:expr, $($t:ident),*; WRITE_TO<$($write_to:ident, $from:ident),*> $op:expr) => {{
        if $device.unified_mem() {
            $crate::to_raw_host!(&$device.cpu, $($t),*);
            $crate::to_raw_host_mut!(&$device.cpu, $($write_to, $from),*);
            $op;

        } else {
            let cpu = $crate::CPU::<$crate::Cached<Base>>::new();
            $crate::cpu_exec_mut!($device, &cpu, $($t),*; WRITE_TO<$($write_to, $from),*> $op);
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

        cl_cpu_exec_unified_mut!(device, buf; WRITE_TO<out, out_cpu> {
            for (out, buf) in out_cpu.iter_mut().zip(buf.iter()) {
                *out += buf + 1;
            }
        });

        assert_eq!(out.read(), [2, 3, 4, 5, 6,]);
    }
}

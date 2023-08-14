//! This module includes macros and functions for executing operations on the CPU.
//! They move the supplied (CUDA, OpenCL, WGPU, ...) `Buffer`s to the CPU and execute the operation on the CPU.
//! Most of the time, you should actually implement the operation for the device natively, as it is typically faster.

#[cfg(feature = "opencl")]
mod cl_may_unified;

#[cfg(feature = "opencl")]
pub use cl_may_unified::*;

use crate::{Alloc, Base, Buffer, Device, Read, Retriever, WriteBuf, CPU};

/// Moves a `Buffer` stored on device `D` to a `CPU` `Buffer`
/// and executes the unary operation `F` with a `CPU` on the newly created `CPU` `Buffer`.
///
/// # Example
#[cfg_attr(feature = "opencl", doc = "```")]
#[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
/// use custos::{exec_on_cpu::cpu_exec_unary, Buffer, Device, OpenCL};
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::new(0)?;
///     
///     let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));
///     
///     let add = 3;
///     
///     let out = cpu_exec_unary(&device, &buf, |cpu, buf| {
///         let mut out = cpu.retrieve(buf.len(), ());
///     
///         for (out, val) in out.iter_mut().zip(buf) {
///             *out += add + val;
///         }
///         out
///     })?;
///     
///     assert_eq!(out.read(), [4, 5, 6, 7, 8]);
///     
///     Ok(())
/// }
/// ```
pub fn cpu_exec_unary<'a, T, D, F>(
    device: &'a D,
    x: &Buffer<T, D>,
    f: F,
) -> crate::Result<Buffer<'a, T, D>>
where
    T: Clone + Default + 'static,
    F: for<'b> Fn(&'b CPU, &Buffer<'_, T, CPU>) -> Buffer<'b, T, CPU>,
    D: Device + Read<T> + WriteBuf<T> + Alloc<T> + Retriever,
{
    let cpu = CPU::<Base>::new();
    let cpu_buf = Buffer::<T, CPU>::from((&cpu, x.read_to_vec()));
    Ok(Buffer::from((device, f(&cpu, &cpu_buf))))
    // TODO add new node to graph
}

/// Moves a single `Buffer` stored on another device to a `CPU` `Buffer`s and executes an operation on the `CPU`.
/// The result is written back to the original `Buffer`.
pub fn cpu_exec_unary_mut<'a, T, D, F>(
    device: &'a D,
    x: &mut Buffer<T, D>,
    f: F,
) -> crate::Result<()>
where
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>),
    D: Read<T> + WriteBuf<T>,
{
    let cpu = CPU::<Base>::new();
    let mut cpu_buf = Buffer::<T, CPU>::from((&cpu, x.read_to_vec()));
    f(&cpu, &mut cpu_buf);

    device.write(x, &cpu_buf);

    Ok(())
}

/// Moves two `Buffer` stored on device `D` to two `CPU` `Buffer`s
/// and executes the binary operation `F` with a `CPU` on the newly created `CPU` `Buffer`s.
///
/// # Example
#[cfg_attr(feature = "opencl", doc = "```")]
#[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
/// use custos::{exec_on_cpu::cpu_exec_binary, Buffer, Device, OpenCL};
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::new(0)?;
///     
///     let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
///     let rhs = Buffer::from((&device, [-1, -4, -1, -8, -1]));
///     
///     let out = cpu_exec_binary(&device, &lhs, &rhs, |cpu, lhs, rhs| {
///         let mut out = cpu.retrieve(lhs.len(), ());
///     
///         for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(&mut out) {
///             *out = lhs + rhs
///         }
///     
///         out
///     });
///     
///     assert_eq!(out.read(), [0, -2, 2, -4, 4]);
///     Ok(())
/// }
/// ```
pub fn cpu_exec_binary<'a, T, D, F>(
    device: &'a D,
    lhs: &Buffer<T, D>,
    rhs: &Buffer<T, D>,
    f: F,
) -> Buffer<'a, T, D>
where
    T: Clone + Default + 'static,
    F: for<'b> Fn(&'b CPU, &Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>) -> Buffer<'b, T, CPU>,
    D: Device + Read<T> + WriteBuf<T> + Alloc<T> + Retriever,
{
    let cpu = CPU::<Base>::new();
    let cpu_lhs = Buffer::<T, CPU>::from((&cpu, lhs.read_to_vec()));
    let cpu_rhs = Buffer::<T, CPU>::from((&cpu, rhs.read_to_vec()));
    Buffer::from((device, f(&cpu, &cpu_lhs, &cpu_rhs)))
    // TODO add new node to graph
}

/// Inplace version of [cpu_exec_binary]
pub fn cpu_exec_binary_mut<T, D, F>(
    device: &D,
    lhs: &mut Buffer<T, D>,
    rhs: &Buffer<T, D>,
    f: F,
) -> crate::Result<()>
where
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>),
    D: Read<T> + WriteBuf<T>,
{
    let cpu = CPU::<Base>::new();
    let mut cpu_lhs = Buffer::<T, CPU>::from((&cpu, lhs.read_to_vec()));
    let cpu_rhs = Buffer::<T, CPU>::from((&cpu, rhs.read_to_vec()));
    f(&cpu, &mut cpu_lhs, &cpu_rhs);

    device.write(lhs, &cpu_lhs);

    Ok(())
}

/// Moves `Buffer`s to `CPU` `Buffer`s.
/// The name of the new `CPU` `Buffer`s are provided by the user.
/// The new `Buffer`s are declared as mutable.
///
/// # Example
#[cfg_attr(feature = "opencl", doc = "```")]
#[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
/// use custos::{CPU, Buffer, OpenCL, to_cpu};
///
/// let device = OpenCL::new(0).unwrap();
///
/// let cpu = CPU::<Base>::new();
///
/// let lhs = Buffer::from((&device, [1, 2, 3]));
/// let rhs = Buffer::from((&device, [1, 2, 3]));
///
/// to_cpu!(cpu, lhs, rhs);
///
/// assert_eq!(lhs.len(), 3);
/// assert_eq!(rhs.len(), 3);
/// ```
#[macro_export]
macro_rules! to_cpu_mut {
    ($cpu:ident, $($t:ident, $cpu_name:ident),*) => {
        $(
            #[allow(unused_mut)]
            let mut $cpu_name = Buffer::<_, CPU>::from((&$cpu, $t.read_to_vec()));
        )*
    };
}

/// Shadows all supplied `Buffer`s to `CPU` `Buffer's.
///
/// # Example
#[cfg_attr(feature = "opencl", doc = "```")]
#[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
/// use custos::{CPU, Buffer, OpenCL, to_cpu};
///
/// let device = OpenCL::new(0).unwrap();
///
/// let cpu = CPU::<Base>::new();
///
/// let lhs = Buffer::from((&device, [1, 2, 3]));
/// let rhs = Buffer::from((&device, [1, 2, 3]));
///
/// to_cpu!(cpu, lhs, rhs);
///
/// assert_eq!(lhs.len(), 3);
/// assert_eq!(rhs.len(), 3);
/// ```
#[macro_export]
macro_rules! to_cpu {
    ($cpu:ident, $($t:ident),*) => {
        $(
            let $t = Buffer::<_, CPU>::from((&$cpu, $t.read_to_vec()));
        )*
    };
}

/// Takes `Buffer`s having a host pointer and wraps them into `CPU` `Buffer`'s.
/// The old `Buffer`s are shadowed.
#[macro_export]
macro_rules! to_raw_host {
    ($($t:ident),*) => {
        $(
            let $t = &unsafe { Buffer::<_, _, ()>::from_raw_host($t.ptr.host_ptr, $t.len()) };
        )*
    };
}

/// Takes `Buffer`s having a host pointer and wraps them into mutable `CPU` `Buffer`'s.
/// New names for the `CPU` `Buffer`s are provided by the user.
#[macro_export]
macro_rules! to_raw_host_mut {
    ($($t:ident, $cpu_name:ident),*) => {
        $(
            let mut $cpu_name = &mut unsafe { Buffer::<_, _, ()>::from_raw_host($t.ptr.host_ptr, $t.len()) };
        )*
    };
}

/// Moves `n` `Buffer`s stored on another device to `n` `CPU` `Buffer`s and executes an operation on the `CPU`.
/// # Example
/* TODO #[cfg_attr(feature = "opencl", doc = "```")]
#[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
/// use custos::{OpenCL, Buffer, CPU};
///
/// let device = OpenCL::new(0).unwrap();
///
/// let a = Buffer::new(&device, 10);
/// let b = Buffer::new(&device, 10);
/// let c = Buffer::new(&device, 10);
///
/// let cpu = CPU::<Base>::new();
///
/// ```
*/
#[macro_export]
macro_rules! cpu_exec {
    ($device:ident, $cpu:ident, $($t:ident),*; $op:expr) => {{
        $crate::to_cpu!($cpu, $($t),*);
        Buffer::from((&$device, $op))
    }};
}

/// Moves `n` `Buffer`s stored on another device to `n` `CPU` `Buffer`s and executes an operation on the `CPU`.
/// The results are written back to the original `Buffer`s.
#[macro_export]
macro_rules! cpu_exec_mut {
    ($device:ident, $cpu:ident, $($t:ident),* WRITE_TO<$($write_to:ident, $from:ident),*> $op:expr) => {{
        $crate::to_cpu!($cpu, $($t),*);
        $crate::to_cpu_mut!($cpu, $($write_to, $from),*);
        $op;
        $(
            $device.write($write_to, &$from);
        )*
    }};
}

/// Moves a single `Buffer` stored on another device to a `CPU` `Buffer` and executes an reduce operation on the `CPU`.
#[inline]
pub fn cpu_exec_reduce<T, D, F>(x: &Buffer<T, D>, f: F) -> T
where
    T: Default + Clone,
    D: Read<T>,
    F: Fn(&CPU, &Buffer<T, CPU>) -> T,
{
    let cpu = CPU::<Base>::new();
    let cpu_x = Buffer::from((&cpu, x.read_to_vec()));
    f(&cpu, &cpu_x)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "opencl")]
    #[test]
    fn test_to_cpu_macro() {
        use crate::{Buffer, CPU};

        let device = crate::OpenCL::new(0).unwrap();

        let cpu = CPU::<Base>::new();

        let lhs = Buffer::from((&device, [1, 2, 3]));
        let rhs = Buffer::from((&device, [1, 2, 3]));

        to_cpu!(cpu, lhs, rhs);
        assert_eq!(lhs.len(), 3);
        assert_eq!(rhs.len(), 3);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_unary_cl() -> crate::Result<()> {
        use crate::{exec_on_cpu::cpu_exec_unary, Buffer, CopySlice, OpenCL};
        let device = OpenCL::new(0)?;

        let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));

        let out = cpu_exec_unary(&device, &buf, |cpu, buf| cpu.copy_slice(buf, ..))?;

        assert_eq!(out.read(), [1, 2, 3, 4, 5]);

        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_unary_cl_unified() -> crate::Result<()> {
        use crate::{exec_on_cpu::cpu_exec_unary_may_unified, Buffer, Device, OpenCL};
        let device = OpenCL::new(0)?;

        let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));

        let add = 3;

        let out = cpu_exec_unary_may_unified(&device, &buf, |cpu, buf| {
            let mut out = cpu.retrieve(buf.len(), ());

            for (out, val) in out.iter_mut().zip(buf) {
                *out += add + val;
            }
            out
        })?;

        assert_eq!(out.read(), [4, 5, 6, 7, 8]);

        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_binary_cl() -> crate::Result<()> {
        use crate::{exec_on_cpu::cpu_exec_binary, Buffer, Device, OpenCL};
        let device = OpenCL::new(0)?;

        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
        let rhs = Buffer::from((&device, [-1, -4, -1, -8, -1]));

        let out = cpu_exec_binary(&device, &lhs, &rhs, |cpu, lhs, rhs| {
            let mut out = cpu.retrieve(lhs.len(), ());

            for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(&mut out) {
                *out = lhs + rhs
            }

            out
        });

        assert_eq!(out.read(), [0, -2, 2, -4, 4]);

        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_binary_cl_may_unified() -> crate::Result<()> {
        use crate::{exec_on_cpu::cpu_exec_binary_may_unified, Buffer, Device, OpenCL};
        let device = OpenCL::new(0)?;

        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
        let rhs = Buffer::from((&device, [-1, -4, -1, -8, -1]));

        let out = cpu_exec_binary_may_unified(&device, &lhs, &rhs, |cpu, lhs, rhs| {
            let mut out = cpu.retrieve(lhs.len(), ());

            for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(&mut out) {
                *out = lhs + rhs
            }

            out
        })?;

        assert_eq!(out.read(), [0, -2, 2, -4, 4]);

        Ok(())
    }
}

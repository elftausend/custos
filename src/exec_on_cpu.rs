//! This module includes macros and functions for executing operations on the CPU.
//! They move the supplied (CUDA, OpenCL, WGPU, ...) `Buffer`s to the CPU and execute the operation on the CPU.
//! Most of the time, you should actually implement the operation for the device natively, as it is typically faster.

#[cfg(feature = "opencl")]
mod cl_may_unified;

#[cfg(feature = "opencl")]
pub use cl_may_unified::*;

use crate::{Alloc, Base, Buffer, Device, Read, Retriever, Unit, WriteBuf, CPU};

/// Moves a `Buffer` stored on device `D` to a `CPU` `Buffer`
/// and executes the unary operation `F` with a `CPU` on the newly created `CPU` `Buffer`.
///
/// # Example
#[cfg_attr(feature = "opencl", doc = "```")]
#[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
/// use custos::{exec_on_cpu::cpu_exec_unary, Buffer, Retriever, OpenCL, Base};
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::<Base>::new(0)?;
///     
///     let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));
///     
///     let add = 3;
///     
///     let out = cpu_exec_unary(&device, &buf, |cpu, buf| {
///         let mut out = cpu.retrieve(buf.len(), ()).unwrap();
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
    T: Unit + Clone + Default + 'static,
    F: for<'b> Fn(&'b CPU, &Buffer<'_, T, CPU>) -> Buffer<'b, T, CPU>,
    D: Read<T> + WriteBuf<T> + Alloc<T> + Retriever<T>,
{
    let cpu = CPU::<Base>::new();
    Ok(crate::cpu_exec!(device, &cpu, x; f(&cpu, &x)))
}

/// Moves a single `Buffer` stored on another device to a `CPU` `Buffer`s and executes an operation on the `CPU`.
/// The result is written back to the original `Buffer`.
pub fn cpu_exec_unary_mut<'a, T, D, F>(
    device: &'a D,
    x: &mut Buffer<T, D>,
    f: F,
) -> crate::Result<()>
where
    T: Unit + Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>),
    D: Read<T> + WriteBuf<T>,
{
    let cpu = CPU::<Base>::new();

    // Works too
    // crate::cpu_exec_mut!(device, &cpu, ; WRITE_TO<x, x_cpu> f(&cpu, &mut x_cpu));
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
/// use custos::{exec_on_cpu::cpu_exec_binary, Buffer, Retriever, OpenCL, Base};
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::<Base>::new(0)?;
///     
///     let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
///     let rhs = Buffer::from((&device, [-1, -4, -1, -8, -1]));
///     
///     let out = cpu_exec_binary(&device, &lhs, &rhs, |cpu, lhs, rhs| {
///         let mut out = cpu.retrieve(lhs.len(), ()).unwrap();
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
    T: Unit + Clone + Default + 'static,
    F: for<'b> Fn(&'b CPU, &Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>) -> Buffer<'b, T, CPU>,
    D: Device + Read<T> + WriteBuf<T> + Alloc<T> + Retriever<T>,
{
    let cpu = CPU::<Base>::new();
    crate::cpu_exec!(device, &cpu, lhs, rhs; f(&cpu, &lhs, &rhs))
}

/// Inplace version of [cpu_exec_binary]
pub fn cpu_exec_binary_mut<T, D, F>(
    device: &D,
    lhs: &mut Buffer<T, D>,
    rhs: &Buffer<T, D>,
    f: F,
) -> crate::Result<()>
where
    T: Unit + Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>),
    D: Read<T> + WriteBuf<T>,
{
    let cpu = CPU::<Base>::new();

    // Should work too
    // crate::cpu_exec_mut!(device, &cpu, rhs; WRITE_TO<lhs, lhs_cpu> f(&cpu, &mut lhs_cpu, &rhs));
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
/// use custos::{CPU, Buffer, OpenCL, to_cpu, Base};
///
/// let device = OpenCL::<Base>::new(0).unwrap();
///
/// let cpu = CPU::<Base>::new();
///
/// let lhs = Buffer::from((&device, [1, 2, 3]));
/// let rhs = Buffer::from((&device, [1, 2, 3]));
///
/// to_cpu!(&cpu, lhs, rhs);
///
/// assert_eq!(lhs.len(), 3);
/// assert_eq!(rhs.len(), 3);
/// ```
#[macro_export]
macro_rules! to_cpu_mut {
    ($cpu:expr, $($t:ident, $cpu_name:ident),*) => {
        $(
            #[allow(unused_mut)]
            let mut $cpu_name = $crate::Buffer::<_, _>::from(($cpu, $t.read_to_vec()));
        )*
    };
}

/// Shadows all supplied `Buffer`s to `CPU` `Buffer's.
///
/// # Example
#[cfg_attr(feature = "opencl", doc = "```")]
#[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
/// use custos::{CPU, Buffer, OpenCL, to_cpu, Base};
///
/// let device = OpenCL::<Base>::new(0).unwrap();
///
/// let cpu = CPU::<Base>::new();
///
/// let lhs = Buffer::from((&device, [1, 2, 3]));
/// let rhs = Buffer::from((&device, [1, 2, 3]));
///
/// to_cpu!(&cpu, lhs, rhs);
///
/// assert_eq!(lhs.len(), 3);
/// assert_eq!(rhs.len(), 3);
/// ```
#[macro_export]
macro_rules! to_cpu {
    ($cpu:expr, $($t:ident),*) => {
        $(
            let $t = $crate::Buffer::<_, _>::from(($cpu, $t.read_to_vec()));
        )*
    };
}

/// Takes `Buffer`s having a host pointer and wraps them into `CPU` `Buffer`'s.
/// The old `Buffer`s are shadowed.
#[macro_export]
macro_rules! to_raw_host {
    ($cpu:expr, $($t:ident),*) => {
        $(
            let $t = &unsafe { $crate::Buffer::<_, _, ()>::from_raw_host_device($cpu, $t.base().host_ptr, $t.len()) };
        )*
    };
}

/// Takes `Buffer`s having a host pointer and wraps them into mutable `CPU` `Buffer`'s.
/// New names for the `CPU` `Buffer`s are provided by the user.
#[macro_export]
macro_rules! to_raw_host_mut {
    ($cpu:expr, $($t:ident, $cpu_name:ident),*) => {
        $(
            #[allow(unused_mut)]
            let mut $cpu_name = &mut unsafe { $crate::Buffer::<_, _, ()>::from_raw_host_device($cpu, $t.base().host_ptr, $t.len()) };
        )*
    };
}

/// Moves `n` `Buffer`s stored on another device to `n` `CPU` `Buffer`s and executes an operation on the `CPU`.
/// # Example
#[cfg_attr(feature = "opencl", doc = "```")]
#[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
/// use custos::{Base, Device, OpenCL, Buffer, CPU, Retriever, opencl::chosen_cl_idx};
///
/// let device = OpenCL::<Base>::new(chosen_cl_idx()).unwrap();
///
/// let a = device.buffer([1, 2, 3, 4]);
/// let b = device.buffer([4, 5, 6, 7]);
///
/// let cpu = CPU::<Base>::new();
/// let c: Buffer<i32, _> = custos::cpu_exec!(&device, &cpu, a, b; {
///     let mut c = cpu.retrieve(a.len(), (&a, &b)).unwrap();
///     for ((a, b), c) in a.iter().zip(&b).zip(c.iter_mut()) {
///         *c = a + b;
///     }
///     c
/// });
///
/// assert_eq!(c.read(), vec![5, 7, 9, 11]);
/// ```
#[macro_export]
macro_rules! cpu_exec {
    ($device:expr, $cpu:expr, $($t:ident),*; $op:expr) => {{
        $crate::to_cpu!($cpu, $($t),*);
        $crate::Buffer::from(($device, $op))
    }};
}

/// Moves `n` `Buffer`s stored on another device to `n` `CPU` `Buffer`s and executes an operation on the `CPU`.
/// The results are written back to the original `Buffer`s.
#[macro_export]
macro_rules! cpu_exec_mut {
    ($device:expr, $cpu:expr, $($t:ident),*; WRITE_TO<$($write_to:ident, $from:ident),*> $op:expr) => {{
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
    T: Unit + Default + Clone,
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
        use crate::{opencl::chosen_cl_idx, Base, Buffer, CPU};

        let device = crate::OpenCL::<Base>::new(chosen_cl_idx()).unwrap();

        let cpu = &CPU::<Base>::new();

        let lhs = Buffer::from((&device, [1, 2, 3]));
        let rhs = Buffer::from((&device, [1, 2, 3]));

        to_cpu!(cpu, lhs, rhs);
        assert_eq!(lhs.len(), 3);
        assert_eq!(rhs.len(), 3);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_exec_cpu_macro() {
        use crate::{prelude::chosen_cl_idx, Base, Device, Retriever};

        let device = crate::OpenCL::<Base>::new(chosen_cl_idx()).unwrap();

        let lhs = device.buffer([1, 2, 3, 4]);
        let rhs = device.buffer([1, 2, 3, 4]);

        let a: crate::Buffer<i32, crate::OpenCL> = cpu_exec!(
            &device, &device.cpu, lhs, rhs; {
                let mut out = device.cpu.retrieve(lhs.len(), (&lhs, &rhs)).unwrap();
                for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(out.iter_mut()) {
                    *out = lhs + rhs;
                }
                out
            }
        );

        assert_eq!(a.read(), vec![2, 4, 6, 8]);
        let cpu = crate::CPU::<Base>::new();
        let other_a: crate::Buffer<i32, crate::OpenCL> = cpu_exec!(
            &device, &cpu, lhs, rhs; {
                let mut out = cpu.retrieve(lhs.len(), (&lhs, &rhs)).unwrap();
                for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(out.iter_mut()) {
                    *out = lhs + rhs;
                }
                out
            }
        );
        assert_eq!(other_a.read(), vec![2, 4, 6, 8]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_unary_cl() -> crate::Result<()> {
        use crate::{
            exec_on_cpu::cpu_exec_unary, opencl::chosen_cl_idx, Base, Buffer, CopySlice, OpenCL,
        };
        let device = OpenCL::<Base>::new(chosen_cl_idx())?;

        let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));

        let out = cpu_exec_unary(&device, &buf, |cpu, buf| cpu.copy_slice(buf, ..))?;

        assert_eq!(out.read(), [1, 2, 3, 4, 5]);

        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_unary_cl_unified() -> crate::Result<()> {
        use crate::{
            exec_on_cpu::cpu_exec_unary_may_unified, opencl::chosen_cl_idx, Base, Buffer, Cached,
            OpenCL, Retriever,
        };
        let device = OpenCL::<Cached<Base>>::new(chosen_cl_idx())?;

        let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));

        let add = 3;

        let out = cpu_exec_unary_may_unified(&device, &buf, |cpu, buf| {
            let mut out = cpu.retrieve(buf.len(), ()).unwrap();

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
        use crate::{
            exec_on_cpu::cpu_exec_binary, opencl::chosen_cl_idx, Base, Buffer, OpenCL, Retriever,
        };
        let device = OpenCL::<Base>::new(chosen_cl_idx())?;

        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
        let rhs = Buffer::from((&device, [-1, -4, -1, -8, -1]));

        let out = cpu_exec_binary(&device, &lhs, &rhs, |cpu, lhs, rhs| {
            let mut out = cpu.retrieve(lhs.len(), ()).unwrap();

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
        use crate::{
            exec_on_cpu::cpu_exec_binary_may_unified, opencl::chosen_cl_idx, Base, Buffer, Cached,
            OpenCL, Retriever,
        };
        let device = OpenCL::<Cached<Base>>::new(chosen_cl_idx())?;

        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
        let rhs = Buffer::from((&device, [-1, -4, -1, -8, -1]));

        let out = cpu_exec_binary_may_unified(&device, &lhs, &rhs, |cpu, lhs, rhs| {
            let mut out = cpu.retrieve(lhs.len(), ()).unwrap();

            for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(&mut out) {
                *out = lhs + rhs
            }

            out
        })?;

        assert_eq!(out.read(), [0, -2, 2, -4, 4]);

        Ok(())
    }

    pub trait AddEw<T, D: crate::Device = Self>: crate::Device {
        fn add(&self, lhs: &crate::Buffer<T, D>, rhs: &crate::Buffer<T, D>) -> crate::Buffer<T, D>;
    }

    impl<Mods, T> AddEw<T> for crate::CPU<Mods>
    where
        Mods: crate::hooks::OnDropBuffer + crate::Retrieve<Self, T> + 'static,
        Self::Base<T, ()>: core::ops::Deref<Target = [T]>,
        T: core::ops::Add<Output = T> + Copy,
    {
        fn add(
            &self,
            lhs: &crate::Buffer<T, Self>,
            rhs: &crate::Buffer<T, Self>,
        ) -> crate::Buffer<T, Self> {
            use crate::Retriever;
            let mut out = self.retrieve(lhs.len(), (lhs, rhs)).unwrap();
            for idx in 0..lhs.len() {
                out[idx] = lhs[idx] + rhs[idx]
            }
            out
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_macro() -> crate::Result<()> {
        use crate::{prelude::chosen_cl_idx, Base, Cached, Device, OpenCL, CPU};

        let device = OpenCL::<Cached<Base>>::new(chosen_cl_idx())?;
        let cpu = CPU::<Cached<Base>>::new();

        let lhs = device.buffer([1, 2, 3, 4, 5]);
        let rhs = device.buffer([-1, -4, -1, -8, -1]);
        
        let out1 = crate::cpu_exec!(&device, &cpu, lhs, rhs; cpu.add(&lhs, &rhs));

        let out = {             
            let lhs = crate::Buffer::<_, _>::from(((&cpu), lhs. read_to_vec()));             
            let rhs = crate::Buffer::<_, _>::from(((&cpu), rhs. read_to_vec()));
            let cpu_out = cpu.add(&lhs, &rhs);
            crate::Buffer::from((&device, cpu_out))         
        };
        assert_eq!(out1.read(), out.read());
        Ok(())
    }
}

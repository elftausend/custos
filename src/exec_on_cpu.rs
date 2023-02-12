#[cfg(feature = "opencl")]
mod cl_may_unified;

#[cfg(feature = "opencl")]
pub use cl_may_unified::*;

use crate::{Alloc, Buffer, Device, Read, WriteBuf, CPU};

/// Moves a `Buffer` stored on device `D` to a `CPU` `Buffer` 
/// and executes the unary operation `F` with a `CPU` on the newly created `CPU` `Buffer`.
/// 
/// # Example
/// ```
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
///         let mut out = cpu.retrieve(buf.len());
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
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &Buffer<'_, T, CPU>) -> Buffer<'b, T, CPU>,
    D: Device + Read<T> + WriteBuf<T> + for<'c> Alloc<'c, T>,
{
    let cpu = CPU::new();
    let cpu_buf = Buffer::<T, CPU>::from((&cpu, x.read_to_vec()));
    Ok(Buffer::from((device, f(&cpu, &cpu_buf))))
}

/// Moves two `Buffer` stored on device `D` to two `CPU` `Buffer`s
/// and executes the binary operation `F` with a `CPU` on the newly created `CPU` `Buffer`s.
/// 
/// # Example
/// ```
/// use custos::{exec_on_cpu::cpu_exec_binary, Buffer, Device, OpenCL};
/// 
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::new(0)?;
///     
///     let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
///     let rhs = Buffer::from((&device, [-1, -4, -1, -8, -1]));
///     
///     let out = cpu_exec_binary(&device, &lhs, &rhs, |cpu, lhs, rhs| {
///         let mut out = cpu.retrieve(lhs.len());
///     
///         for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(&mut out) {
///             *out = lhs + rhs
///         }
///     
///         out
///     })?;
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
) -> crate::Result<Buffer<'a, T, D>>
where
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>) -> Buffer<'b, T, CPU>,
    D: Device + Read<T> + WriteBuf<T> + for<'c> Alloc<'c, T>,
{
    let cpu = CPU::new();
    let cpu_lhs = Buffer::<T, CPU>::from((&cpu, lhs.read_to_vec()));
    let cpu_rhs = Buffer::<T, CPU>::from((&cpu, rhs.read_to_vec()));
    Ok(Buffer::from((device, f(&cpu, &cpu_lhs, &cpu_rhs))))
}

pub fn cpu_exec_binary_mut<'a, T, D, F>(
    device: &'a D,
    lhs: &mut Buffer<T, D>,
    rhs: &Buffer<T, D>,
    f: F,
) -> crate::Result<()>
where
    T: Clone + Default,
    F: for<'b> Fn(&'b CPU, &mut Buffer<'_, T, CPU>, &Buffer<'_, T, CPU>),
    D: Device + Read<T> + WriteBuf<T> + for<'c> Alloc<'c, T>,
{
    let cpu = CPU::new();
    let mut cpu_lhs = Buffer::<T, CPU>::from((&cpu, lhs.read_to_vec()));
    let cpu_rhs = Buffer::<T, CPU>::from((&cpu, rhs.read_to_vec()));
    f(&cpu, &mut cpu_lhs, &cpu_rhs);

    device.write(lhs, &cpu_lhs);

    Ok(())
}

#[cfg(test)]
mod tests {

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
            let mut out = cpu.retrieve(buf.len());

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
            let mut out = cpu.retrieve(lhs.len());

            for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(&mut out) {
                *out = lhs + rhs
            }

            out
        })?;

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
            let mut out = cpu.retrieve(lhs.len());

            for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(&mut out) {
                *out = lhs + rhs
            }

            out
        })?;

        assert_eq!(out.read(), [0, -2, 2, -4, 4]);

        Ok(())
    }
}
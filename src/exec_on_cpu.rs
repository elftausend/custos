#[cfg(feature = "opencl")]
mod cl_may_unified;

#[cfg(feature = "opencl")]
pub use cl_may_unified::*;

use crate::{Buffer, Device, Read, CPU, WriteBuf, Alloc};

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

#[cfg(test)]
mod tests {

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_unary_cl() -> crate::Result<()> {
        use crate::{OpenCL, Buffer, exec_on_cpu::cpu_exec_unary, CopySlice};
        let device = OpenCL::new(0)?;

        let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));

        let out = cpu_exec_unary(&device, &buf, |cpu, buf| cpu.copy_slice(buf, ..))?;

        assert_eq!(out.read(), [1, 2, 3, 4, 5]);

        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cpu_exec_unary_cl_unified() -> crate::Result<()> {
        use crate::{OpenCL, Buffer, exec_on_cpu::cpu_exec_unary_may_unified, Device};
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
}
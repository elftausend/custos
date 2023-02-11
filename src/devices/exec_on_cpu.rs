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

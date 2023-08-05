use crate::{module_comb::{MainMemory, WriteBuf, Buffer, OnDropBuffer}, Shape};

use super::CPU;

impl<Mods: OnDropBuffer, T: Copy, D: MainMemory, S: Shape> WriteBuf<T, S, D> for CPU<Mods> {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, D, S>, data: &[T]) {
        buf.copy_from_slice(data)
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, D, S>, src: &Buffer<T, D, S>) {
        self.write(dst, src)
    }
}

#[cfg(test)]
mod tests {
    use crate::module_comb::{CPU, Base, Cached, Buffer, WriteBuf};

    #[test]
    fn test_same_core_device_different_modules() {
        let dev1 = CPU::<Base>::new();
        let dev2 = CPU::<Cached<Base>>::new();

        let mut buf_from_dev2 = Buffer::<_, _, ()>::new(&dev2, 10); 
        dev1.write(&mut buf_from_dev2, &[1, 2, 3, 4,])
    }
}

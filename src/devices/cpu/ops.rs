use core::ops::{Index, RangeBounds};

use crate::{Buffer, ClearBuf, CopySlice, MainMemory, Read, Shape, WriteBuf, CPU};

impl<T, D: MainMemory, S: Shape> Read<T, S, D> for CPU {
    type Read<'a> = &'a [T] where T: 'a, D: 'a, S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, D, S>) -> Self::Read<'a> {
        buf.as_slice()
    }

    #[inline]
    fn read_to_vec<'a>(&self, buf: &Buffer<T, D, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        buf.to_vec()
    }
}

impl<T: Copy, D: MainMemory, S: Shape> WriteBuf<T, S, D> for CPU {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, D, S>, data: &[T]) {
        buf.copy_from_slice(data)
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, D, S>, src: &Buffer<T, D, S>) {
        self.write(dst, src)
    }
}

impl<T: Copy, R: RangeBounds<usize>, D: MainMemory> CopySlice<T, R, D> for CPU
where
    [T]: Index<R, Output = [T]>,
{
    fn copy_slice(&self, buf: &Buffer<T, D>, range: R) -> Buffer<T, Self> {
        let slice = &buf.as_slice()[range];
        let mut copied = Buffer::new(self, slice.len());
        self.write(&mut copied, slice);
        copied
    }
}

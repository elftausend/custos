use core::ops::{Index, Range, RangeBounds};

use crate::{bounds_to_range, Buffer, CopySlice, MainMemory, Read, Shape, WriteBuf, CPU, ClearBuf};

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

// #[impl_stack]
impl<T: Default, D: MainMemory, S: Shape> ClearBuf<T, S, D> for CPU {
    fn clear(&self, buf: &mut Buffer<T, D, S>) {
        for value in buf {
            *value = T::default();
        }
    }
}

impl<T: Copy, D: MainMemory> CopySlice<T, D> for CPU
where
    [T]: Index<Range<usize>, Output = [T]>,
{
    fn copy_slice_to<SR: RangeBounds<usize>, DR: RangeBounds<usize>>(
        &self,
        source: &Buffer<T, D>,
        source_range: SR,
        dest: &mut Buffer<T, Self>,
        dest_range: DR,
    ) {
        let source_range = bounds_to_range(source_range, source.len());
        let dest_range = bounds_to_range(dest_range, dest.len());

        assert_eq!(
            source_range.end - source_range.start,
            dest_range.end - dest_range.start,
        );

        dest[dest_range].copy_from_slice(&source[source_range]);
    }

    fn copy_slice_all<I: IntoIterator<Item = (Range<usize>, Range<usize>)>>(
        &self,
        source: &Buffer<T, D>,
        dest: &mut Buffer<T, Self>,
        ranges: I,
    ) {
        for (source_range, dest_range) in ranges {
            self.copy_slice_to(source, source_range, dest, dest_range);
        }
    }
}

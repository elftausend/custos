use core::ops::{Bound, RangeBounds};

use crate::{cuda::api::cu_read, Buffer, CDatatype, ClearBuf, CopySlice, Read, WriteBuf, CUDA};

use super::{
    api::{cuMemcpy, cu_write},
    cu_clear,
};

impl<T: Default + Clone> Read<T> for CUDA {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        CUDA: 'a;

    #[inline]
    fn read(&self, buf: &Buffer<T, CUDA>) -> Vec<T> {
        self.read_to_vec(buf)
    }

    fn read_to_vec(&self, buf: &Buffer<T, CUDA>) -> Vec<T>
    where
        T: Default + Clone,
    {
        assert!(
            buf.ptrs().2 != 0,
            "called Read::read(..) on a non CUDA buffer"
        );
        // TODO: sync here or somewhere else?
        self.stream().sync().unwrap();

        let mut read = vec![T::default(); buf.len()];
        cu_read(&mut read, buf.ptr.ptr).unwrap();
        read
    }
}

impl<T: CDatatype> ClearBuf<T> for CUDA {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, CUDA>) {
        cu_clear(self, buf).unwrap()
    }
}

impl<T, R: RangeBounds<usize>> CopySlice<T, R> for CUDA {
    fn copy_slice(&self, buf: &Buffer<T, CUDA>, range: R) -> Buffer<T, Self> {
        let start = match range.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Excluded(end) => *end,
            Bound::Included(end) => end + 1,
            Bound::Unbounded => buf.len(),
        };

        let slice_len = end - start;
        let copied = Buffer::new(self, slice_len);

        unsafe {
            cuMemcpy(
                copied.ptr.ptr,
                buf.ptr.ptr + (start * std::mem::size_of::<T>()) as u64,
                copied.len() * std::mem::size_of::<T>(),
            );
        }

        copied
    }
}

impl<T> WriteBuf<T> for CUDA {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, CUDA>, data: &[T]) {
        cu_write(buf.cu_ptr(), data).unwrap();
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self, ()>, src: &Buffer<T, Self, ()>) {
        unsafe {
            cuMemcpy(
                dst.ptr.ptr,
                src.ptr.ptr,
                src.len() * std::mem::size_of::<T>(),
            );
        }
    }
}

use core::ops::{Range, RangeBounds};

use crate::{
    bounds_to_range,
    cuda::api::{cu_read_async, CUstreamCaptureStatus},
    Buffer, CDatatype, ClearBuf, CopySlice, Read, WriteBuf, CUDA,
};

use super::{
    api::{cuMemcpy, cu_write_async},
    cu_clear,
};

impl<Mods, T: Default + Clone> Read<T> for CUDA<Mods> {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        CUDA<Mods>: 'a;

    #[inline]
    fn read(&self, buf: &Buffer<T, Self>) -> Vec<T> {
        self.read_to_vec(buf)
    }

    fn read_to_vec(&self, buf: &Buffer<T, Self>) -> Vec<T>
    where
        T: Default + Clone,
    {
        assert!(
            buf.ptrs().2 != 0,
            "called Read::read(..) on a non CUDA buffer"
        );
        // TODO: sync here or somewhere else?
        if self.stream().capture_status().unwrap()
            == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            self.stream().sync().unwrap();
        }

        let mut read = vec![T::default(); buf.len()];
        cu_read_async(&mut read, buf.data.ptr, &self.mem_transfer_stream).unwrap();
        self.mem_transfer_stream.sync().unwrap();
        read
    }
}

impl<Mods, T: CDatatype> ClearBuf<T> for CUDA<Mods> {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, Self>) {
        cu_clear(self, buf).unwrap()
    }
}

impl<Mods, T> CopySlice<T> for CUDA<Mods> {
    fn copy_slice_to<SR: RangeBounds<usize>, DR: RangeBounds<usize>>(
        &self,
        source: &Buffer<T, Self>,
        source_range: SR,
        dest: &mut Buffer<T, Self>,
        dest_range: DR,
    ) {
        let source_range = bounds_to_range(source_range, source.len());
        let dest_range = bounds_to_range(dest_range, dest.len());

        let len = source_range.end - source_range.start;
        assert_eq!(len, dest_range.end - dest_range.start);
        let size = std::mem::size_of::<T>();

        unsafe {
            cuMemcpy(
                dest.data.ptr + (dest_range.start * size) as u64,
                source.data.ptr + (source_range.start * size) as u64,
                len * size,
            );
        }
    }

    fn copy_slice_all<I: IntoIterator<Item = (Range<usize>, Range<usize>)>>(
        &self,
        source: &Buffer<T, Self>,
        dest: &mut Buffer<T, Self>,
        ranges: I,
    ) {
        for (source_range, dest_range) in ranges {
            self.copy_slice_to(source, source_range, dest, dest_range);
        }
    }
}

impl<Mods, T> WriteBuf<T> for CUDA<Mods> {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, Self>, data: &[T]) {
        cu_write_async(buf.cu_ptr(), data, &self.mem_transfer_stream).unwrap();
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self, ()>, src: &Buffer<T, Self, ()>) {
        unsafe {
            cuMemcpy(
                dst.data.ptr,
                src.data.ptr,
                src.len() * std::mem::size_of::<T>(),
            );
        }
    }
}

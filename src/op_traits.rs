use core::ops::{Bound, Range, RangeBounds};

use crate::{shape::Shape, Alloc, Buffer, Device, OnDropBuffer, OnNewBuffer};

/// Trait for implementing the clear() operation for the compute devices.
pub trait ClearBuf<T, S: Shape = (), D: Device = Self> {
    /// Sets all elements of the matrix to zero.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, ClearBuf, Buffer};
    ///
    /// let device = CPU::new();
    /// let mut a = Buffer::from((&device, [2, 4, 6, 8, 10, 12]));
    /// assert_eq!(a.read(), vec![2, 4, 6, 8, 10, 12]);
    ///
    /// device.clear(&mut a);
    /// assert_eq!(a.read(), vec![0; 6]);
    /// ```
    fn clear(&self, buf: &mut Buffer<T, D, S>);
}

/// Trait for copying a slice of a buffer, to implement the slice() operation.
pub trait CopySlice<T, D: Device = Self>: Sized + Device {
    /// Copy a slice of the given buffer into a new buffer.
    /// # Example
    ///
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, CopySlice};
    ///
    /// let device = CPU::new();
    /// let buf = Buffer::from((&device, [1., 2., 6., 2., 4.,]));
    /// let slice = device.copy_slice(&buf, 1..3);
    /// assert_eq!(slice.read(), &[2., 6.]);
    /// ```
    fn copy_slice<'a, R: RangeBounds<usize>>(
        &'a self,
        buf: &Buffer<T, D>,
        range: R,
    ) -> Buffer<'a, T, Self>
    where
        Self: Alloc<T> + OnDropBuffer + OnNewBuffer<T, Self, ()>,
    {
        let range = bounds_to_range(range, buf.len());
        let mut copied = Buffer::new(self, range.end - range.start);
        self.copy_slice_to(buf, range, &mut copied, ..);
        copied
    }

    /// Copy a slice of the source buffer into a slice of the destination buffer.
    /// # Example
    ///
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, CopySlice};
    ///
    /// let device = CPU::new();
    /// let source = Buffer::from((&device, [1., 2., 3., 4., 5.,]));
    /// let mut dest = Buffer::from((&device, [5., 4., 3., 2., 1.,]));
    /// let slice = device.copy_slice_to(&source, 1..3, &mut dest, 3..5);
    /// assert_eq!(dest.read(), &[5., 4., 3., 2., 3.]);
    /// ```
    fn copy_slice_to<SR: RangeBounds<usize>, DR: RangeBounds<usize>>(
        &self,
        source: &Buffer<T, D>,
        source_range: SR,
        dest: &mut Buffer<T, Self>,
        dest_range: DR,
    );

    /// Copy multiple slices of the source buffer into multiplie slices of the destination buffer.
    ///
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{Buffer, CPU, CopySlice};
    ///
    /// let device = CPU::new();
    /// let source = Buffer::from((&device, [1., 2., 6., 2., 4.]));
    ///
    /// let mut dest = Buffer::new(&device, 10);
    ///
    /// device.copy_slice_all(&source, &mut dest, [(2..5, 7..10), (1..3, 3..5)]);
    ///
    /// assert_eq!(
    ///    dest.read(),
    ///    [0.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 6.0, 2.0, 4.0]
    /// );
    ///```
    fn copy_slice_all<I: IntoIterator<Item = (Range<usize>, Range<usize>)>>(
        &self,
        source: &Buffer<T, D>,
        dest: &mut Buffer<T, Self>,
        ranges: I,
    );
}

/// Trait for reading buffers.
/// Syncronizationpoint for CUDA.
pub trait Read<T, S: Shape = (), D: Device = Self>: Device {
    /// The type of the read data.
    /// Usually `Vec<T>` or `&'a [T]`.
    type Read<'a>
    where
        T: 'a,
        D: 'a,
        S: 'a;

    /// Read the data of the `Buffer` as type `Read`.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, Read};
    ///
    /// let device = CPU::new();
    /// let a = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    /// let read = device.read(&a);
    /// assert_eq!(&[1., 2., 3., 3., 2., 1.,], read);
    /// ```
    fn read<'a>(&self, buf: &'a Buffer<T, D, S>) -> Self::Read<'a>;

    /// Read the data of a buffer into a vector
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, Read};
    ///
    /// let device = CPU::new();
    /// let a = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    /// let read = device.read_to_vec(&a);
    /// assert_eq!(vec![1., 2., 3., 3., 2., 1.,], read);
    /// ```
    #[cfg(not(feature = "no-std"))]
    fn read_to_vec(&self, buf: &Buffer<T, D, S>) -> Vec<T>
    where
        T: Default + Clone;
}

/// Trait for writing data to buffers.
pub trait WriteBuf<T, S: Shape = (), D: Device = Self>: Device {
    /// Write data to the buffer.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, WriteBuf};
    ///
    /// let device = CPU::new();
    /// let mut buf: Buffer<i32> = Buffer::new(&device, 4);
    /// device.write(&mut buf, &[9, 3, 2, -4]);
    /// assert_eq!(buf.as_slice(), &[9, 3, 2, -4])
    ///
    /// ```
    fn write(&self, buf: &mut Buffer<T, D, S>, data: &[T]);

    /// Writes data from `<Device>` Buffer to other `<Device>` Buffer.
    /// The buffers must have the same size.
    ///
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, WriteBuf};
    ///
    /// let device = CPU::new();
    ///
    /// let mut dst: Buffer<i32> = Buffer::new(&device, 4);
    ///
    /// let mut src: Buffer<i32> = Buffer::from((&device, [1, 2, -5, 4]));
    /// device.write_buf(&mut dst, &src);
    /// assert_eq!(dst.read(), [1, 2, -5, 4])
    /// ```
    fn write_buf(&self, dst: &mut Buffer<T, D, S>, src: &Buffer<T, D, S>);
}

/// This trait is used to clone a buffer based on a specific device type.
pub trait CloneBuf<'a, T, S: Shape = ()>: Sized + Device {
    /// Creates a deep copy of the specified buffer.
    /// # Example
    ///
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, CloneBuf};
    ///
    /// let device = CPU::new();
    /// let buf = Buffer::from((&device, [1., 2., 6., 2., 4.,]));
    ///
    /// let cloned = device.clone_buf(&buf);
    /// assert_eq!(buf.read(), cloned.read());
    /// ```
    fn clone_buf(&'a self, buf: &Buffer<'a, T, Self, S>) -> Buffer<'a, T, Self, S>;
}

/// Convert a possibly-indefinite [`RangeBounds`] into a [`Range`] with a start and stop index.
#[inline]
pub(crate) fn bounds_to_range<B: RangeBounds<usize>>(bounds: B, len: usize) -> Range<usize> {
    let start = match bounds.start_bound() {
        Bound::Included(start) => *start,
        Bound::Excluded(start) => start + 1,
        Bound::Unbounded => 0,
    };

    let end = match bounds.end_bound() {
        Bound::Excluded(end) => *end,
        Bound::Included(end) => end + 1,
        Bound::Unbounded => len,
    };

    start..end
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "stack")]
    #[cfg(feature = "macro")]
    #[cfg(not(feature = "autograd"))]
    #[test]
    fn test_unary_ew_stack_no_autograd() {
        use crate::{Buffer, Combiner, Dim1, UnaryElementWiseMayGrad};

        let device = crate::Stack::new();
        let buf = Buffer::<_, _, Dim1<5>>::from((&device, [1, 2, 4, 5, 3]));

        let out = device.unary_ew(&buf, |x| x.mul(3), |x| x);

        assert_eq!(out.read(), [3, 6, 12, 15, 9]);
    }
}

use core::ops::RangeBounds;


use crate::{Buffer, Device, Shape};

/// Trait for implementing the clear() operation for the compute devices.
pub trait ClearBuf<T, D: Device = Self, S: Shape = ()>: Device {
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
    fn copy_slice<R: RangeBounds<usize>>(&self, buf: &Buffer<T, D>, range: R) -> Buffer<T, Self>;
}

/// Trait for reading buffers.
pub trait Read<T, D: Device = Self, S: Shape = ()>: Device {
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
pub trait WriteBuf<T, D: Device = Self, S: Shape = ()>: Sized + Device {
    /// Write data to the buffer.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, WriteBuf};
    ///
    /// let device = CPU::new();
    /// let mut buf = Buffer::new(&device, 4);
    /// device.write(&mut buf, &[9, 3, 2, -4]);
    /// assert_eq!(buf.as_slice(), &[9, 3, 2, -4])
    ///
    /// ```
    fn write(&self, buf: &mut Buffer<T, D, S>, data: &[T]);
    /// Writes data from <Device> Buffer to other <Device> Buffer.
    // TODO: implement, change name of fn? -> set_.. ?
    fn write_buf(&self, _dst: &mut Buffer<T, Self, S>, _src: &Buffer<T, Self, S>) {
        unimplemented!()
    }
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

/// This trait is used to retrieve a cached buffer from a specific device type.
pub trait CacheBuf<'a, T, S: Shape = ()>: Sized + Device {
    /// Adds a buffer to the cache. Following calls will return this buffer, if the corresponding internal count matches with the id used in the cache.
    /// # Example
    #[cfg_attr(any(feature = "realloc", not(feature="cpu")), doc = "```ignore")]
    #[cfg_attr(any(not(feature = "realloc"), feature="cpu"), doc = "```")]
    /// use custos::{CPU, Read, set_count, get_count, CacheBuf};
    ///
    /// let device = CPU::new();
    /// assert_eq!(0, get_count());
    ///
    /// let mut buf = CacheBuf::<f32>::cached(&device, 10);
    /// assert_eq!(1, get_count());
    ///
    /// for value in buf.as_mut_slice() {
    ///     *value = 1.5;
    /// }
    ///    
    /// set_count(0);
    /// let buf = CacheBuf::<f32>::cached(&device, 10);
    /// assert_eq!(device.read(&buf), vec![1.5; 10]);
    /// ```
    fn cached(&'a self, len: usize) -> Buffer<'a, T, Self, S>;
}

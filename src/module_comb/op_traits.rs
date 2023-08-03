use crate::Shape;

use super::{Device, Buffer};


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

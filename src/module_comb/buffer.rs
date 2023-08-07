use super::{Alloc, Base, Device, HasId, MainMemory, OnNewBuffer, WriteBuf, CPU};
use crate::{flag::AllocFlag, PtrType, Shape};

pub struct Buffer<'a, T = f32, D: Device = CPU<Base>, S: Shape = ()> {
    /// the type of pointer
    pub data: D::Data<T, S>,
    /// A reference to the corresponding device. Mainly used for operations without a device parameter.
    pub device: Option<&'a D>,
}

impl<'a, T, D: Device, S: Shape> Buffer<'a, T, D, S> {
    #[inline]
    pub fn new(device: &'a D, len: usize) -> Self
    where
        D: OnNewBuffer<T, D, S>,
    {
        let data = device.alloc(len, crate::flag::AllocFlag::None);
        Buffer::from_new_alloc(device, data)
    }

    #[inline]
    fn from_new_alloc(device: &'a D, data: D::Data<T, S>) -> Self
    where
        D: OnNewBuffer<T, D, S>,
    {
        let buf = Buffer {
            data,
            device: Some(device),
        };

        // mind: on_new_buffer must be called for user buffers!
        device.on_new_buffer(device, &buf);
        buf
    }

    #[inline]
    pub fn device(&self) -> &'a D {
        self.device
            .expect("Called device() on a deviceless buffer.")
    }

    /// Returns the number of elements contained in `Buffer`.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::module_comb::{Base, CPU, Buffer};
    ///
    /// let device = CPU::<Base>::new();
    /// let a = Buffer::<i32, _>::new(&device, 10);
    /// assert_eq!(a.len(), 10)
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.data.size()
    }

    /// Writes a slice to the `Buffer`.
    /// With a CPU buffer, the slice is just copied to the slice of the buffer.
    ///
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::module_comb::{CPU, Buffer, Base};
    ///
    /// let device = CPU::<Base>::new();
    /// let mut buf = Buffer::<i32>::new(&device, 6);
    /// buf.write(&[4, 2, 3, 4, 5, 3]);
    ///
    /// assert_eq!(&*buf, [4, 2, 3, 4, 5, 3]);
    /// ```
    #[inline]
    pub fn write(&mut self, data: &[T])
    where
        D: WriteBuf<T, S, D>,
    {
        self.device().write(self, data)
    }

    /// Writes the contents of the source buffer to self.
    #[inline]
    pub fn write_buf(&mut self, src: &Buffer<T, D, S>)
    where
        T: Clone,
        D: WriteBuf<T, S, D>,
    {
        self.device().write_buf(self, src)
    }
}

impl<'a, T, D: Device, S: Shape> HasId for Buffer<'a, T, D, S> {
    #[inline]
    fn id(&self) -> super::Id {
        self.data.id()
    }
}

impl<'a, T, D: Device, S: Shape> Drop for Buffer<'a, T, D, S> {
    #[inline]
    fn drop(&mut self) {
        if self.data.flag() != AllocFlag::None {
            return;
        }

        if let Some(device) = self.device {
            device.on_drop_buffer(device, self)
        }
    }
}

impl<'a, T, D: Device + OnNewBuffer<T, D, S>, S: Shape> Buffer<'a, T, D, S> {
    /// Creates a new `Buffer` from a slice (&[T]).
    #[inline]
    pub fn from_slice(device: &'a D, slice: &[T]) -> Self
    where
        T: Clone,
        D: Alloc,
    {
        let data = device.alloc_from_slice(slice);
        Buffer::from_new_alloc(device, data)
    }

    /// Creates a new `Buffer` from a `Vec`.
    #[cfg(not(feature = "no-std"))]
    #[inline]
    pub fn from_vec(device: &'a D, data: Vec<T>) -> Self
    where
        T: Clone,
        D: Alloc,
    {
        let data = device.alloc_from_vec(data);
        Buffer::from_new_alloc(device, data)
    }

    /// Creates a new `Buffer` from an nd-array.
    /// The dimension is defined by the [`Shape`].
    #[inline]
    pub fn from_array(device: &'a D, array: S::ARR<T>) -> Buffer<T, D, S>
    where
        T: Clone,
        D: Alloc,
    {
        let data = device.alloc_from_array(array);
        Buffer::from_new_alloc(device, data)
    }
}

/// A `Buffer` dereferences into a slice.
///
/// # Examples
///
#[cfg_attr(feature = "cpu", doc = "```")]
#[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
/// use custos::module_comb::{Base, Buffer, CPU};
///
/// let device = CPU::<Base>::new();
///
/// let a = Buffer::from((&device, [1., 2., 3., 4.,]));
/// let b = Buffer::from((&device, [2., 3., 4., 5.,]));
///
/// let mut c = Buffer::from((&device, [0.; 4]));
///
/// let slice_add = |a: &[f64], b: &[f64], c: &mut [f64]| {
///     for i in 0..c.len() {
///         c[i] = a[i] + b[i];
///     }
/// };
///
/// slice_add(&a, &b, &mut c);
/// assert_eq!(c.as_slice(), &[3., 5., 7., 9.,]);
/// ```
impl<T, D: MainMemory, S: Shape> core::ops::Deref for Buffer<'_, T, D, S> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { core::slice::from_raw_parts(D::as_ptr(&self.data), self.len()) }
    }
}

/// A `Buffer` dereferences into a mutable slice.
///
/// # Examples
///
#[cfg_attr(feature = "cpu", doc = "```")]
#[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
/// use custos::module_comb::{Base, Buffer, CPU};
///  
/// let device = CPU::<Base>::new();
///
/// let a = Buffer::from((&device, [4., 2., 3., 4.,]));
/// let b = Buffer::from((&device, [2., 3., 6., 5.,]));
/// let mut c = Buffer::from((&device, [0.; 4]));
///
/// let slice_add = |a: &[f64], b: &[f64], c: &mut [f64]| {
///     for i in 0..c.len() {
///         c[i] = a[i] + b[i];
///     }
/// };
/// slice_add(&a, &b, &mut c);
/// assert_eq!(c.as_slice(), &[6., 5., 9., 9.,]);
/// ```
impl<T, D: MainMemory, S: Shape> core::ops::DerefMut for Buffer<'_, T, D, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(D::as_ptr_mut(&mut self.data), self.len()) }
    }
}

#[cfg(test)]
mod tests {}

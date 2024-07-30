use crate::{prelude::Number, shape::Shape, Alloc, Buffer, Dim1, Dim2, Dim3, OnNewBuffer, Unit};

/// Trait for creating [`Buffer`]s with a [`Shape`]. The [`Shape`] is inferred from the array.
pub trait WithShape<'a, D, C> {
    /// Create a new [`Buffer`] with the given [`Shape`] and array. The [`Shape`] is typically inferred from the ND-array.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, WithShape, Base};
    ///
    /// let device = CPU::<Base>::new();
    /// let buf = Buffer::with(&device, [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(&**buf, &[1.0, 2.0, 3.0]);
    ///
    /// ```
    fn with(device: &'a D, array: C) -> Self;
}

impl<'a, T, D, const N: usize> WithShape<'a, D, [T; N]> for Buffer<'a, T, D, Dim1<N>>
where
    T: Number, // using Number here, because T could be an array type
    D: Alloc<T> + OnNewBuffer<'a, T, D, Dim1<N>>,
{
    #[inline]
    fn with(device: &'a D, array: [T; N]) -> Self {
        Buffer::from_array(device, array)
    }
}

impl<'a, T, D, const N: usize> WithShape<'a, D, &[T; N]> for Buffer<'a, T, D, Dim1<N>>
where
    T: Number,
    D: Alloc<T> + OnNewBuffer<'a, T, D, Dim1<N>>,
{
    #[inline]
    fn with(device: &'a D, array: &[T; N]) -> Self {
        Buffer::from_array(device, *array)
    }
}

impl<'a, T, D, const B: usize, const A: usize> WithShape<'a, D, [[T; A]; B]>
    for Buffer<'a, T, D, Dim2<B, A>>
where
    T: Number,
    D: Alloc<T> + OnNewBuffer<'a, T, D, Dim2<B, A>>,
{
    #[inline]
    fn with(device: &'a D, array: [[T; A]; B]) -> Self {
        Buffer::from_array(device, array)
    }
}

impl<'a, T, D, const C: usize, const B: usize, const A: usize> WithShape<'a, D, [[[T; A]; B]; C]>
    for Buffer<'a, T, D, Dim3<C, B, A>>
where
    T: Number,
    D: Alloc<T> + OnNewBuffer<'a, T, D, Dim3<C, B, A>>,
{
    #[inline]
    fn with(device: &'a D, array: [[[T; A]; B]; C]) -> Self {
        Buffer::from_array(device, array)
    }
}

impl<'a, T, D, const C: usize, const B: usize, const A: usize> WithShape<'a, D, &[[[T; A]; B]; C]>
    for Buffer<'a, T, D, Dim3<C, B, A>>
where
    T: Number,
    D: Alloc<T> + OnNewBuffer<'a, T, D, Dim3<C, B, A>>,
{
    #[inline]
    fn with(device: &'a D, array: &[[[T; A]; B]; C]) -> Self {
        Buffer::from_array(device, *array)
    }
}

impl<'a, T, D, S: Shape> WithShape<'a, D, ()> for Buffer<'a, T, D, S>
where
    T: Unit,
    D: Alloc<T> + OnNewBuffer<'a, T, D, S>,
{
    fn with(device: &'a D, _: ()) -> Self {
        Buffer::new(device, S::LEN)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    #[test]
    fn test_with_const_dim2_cpu() {
        use crate::{Base, Buffer, WithShape, CPU};

        let device = CPU::<Base>::new();

        let buf = Buffer::with(&device, [[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(&**buf, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(feature = "stack")]
    #[test]
    fn test_with_const_dim2_stack() {
        use crate::{Base, Buffer, Stack, WithShape};

        let device = Stack::<Base>::new();

        let buf = Buffer::with(&device, [[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(buf.data.array, [[1.0, 2.0,], [3.0, 4.0]]);
    }
}

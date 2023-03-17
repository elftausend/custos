use crate::{prelude::Number, shape::Shape, Alloc, Buffer, Dim1, Dim2, Ident};

/// Trait for creating [`Buffer`]s with a [`Shape`]. The [`Shape`] is inferred from the array.
pub trait WithShape<D, C> {
    /// Create a new [`Buffer`] with the given [`Shape`] and array.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, WithShape};
    ///
    /// let device = CPU::new();
    /// let buf = Buffer::with(&device, [1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(&*buf, &[1.0, 2.0, 3.0]);
    ///
    /// ```
    fn with(device: D, array: C) -> Self;
}

impl<'a, T, D, const N: usize> WithShape<&'a D, [T; N]> for Buffer<'a, T, D, Dim1<N>>
where
    T: Number, // using Number here, because T could be an array type
    D: Alloc<'a, T, Dim1<N>>,
{
    fn with(device: &'a D, array: [T; N]) -> Self {
        Buffer {
            ident: Ident::new_bumped(array.len()),
            ptr: device.with_array(array),
            device: Some(device),
        }
    }
}

impl<'a, T, D, const N: usize> WithShape<&'a D, &[T; N]> for Buffer<'a, T, D, Dim1<N>>
where
    T: Number,
    D: Alloc<'a, T, Dim1<N>>,
{
    fn with(device: &'a D, array: &[T; N]) -> Self {
        Buffer {
            ident: Ident::new_bumped(array.len()),
            ptr: device.with_array(*array),
            device: Some(device),
        }
    }
}

impl<'a, T, D, const B: usize, const A: usize> WithShape<&'a D, [[T; A]; B]>
    for Buffer<'a, T, D, Dim2<B, A>>
where
    T: Number,
    D: Alloc<'a, T, Dim2<B, A>>,
{
    fn with(device: &'a D, array: [[T; A]; B]) -> Self {
        Buffer {
            ident: Ident::new_bumped(B * A),
            ptr: device.with_array(array),
            device: Some(device),
        }
    }
}

impl<'a, T, D, const B: usize, const A: usize> WithShape<&'a D, &[[T; A]; B]>
    for Buffer<'a, T, D, Dim2<B, A>>
where
    T: Number,
    D: Alloc<'a, T, Dim2<B, A>>,
{
    fn with(device: &'a D, array: &[[T; A]; B]) -> Self {
        Buffer {
            ident: Ident::new_bumped(B * A),
            ptr: device.with_array(*array),
            device: Some(device),
        }
    }
}

impl<'a, T, D, S: Shape> WithShape<&'a D, ()> for Buffer<'a, T, D, S>
where
    D: Alloc<'a, T, S>,
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
        use crate::{Buffer, WithShape, CPU};

        let device = CPU::new();

        let buf = Buffer::with(&device, [[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(&*buf, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(feature = "stack")]
    #[test]
    fn test_with_const_dim2_stack() {
        use crate::{Buffer, Stack, WithShape};

        let device = Stack;

        let buf = Buffer::with(&device, [[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(buf.ptr.array, [[1.0, 2.0,], [3.0, 4.0]]);
    }
}

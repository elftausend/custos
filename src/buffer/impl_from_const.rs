use crate::{shape::Shape, Alloc, Buffer, Dim1, Dim2, prelude::Number};

pub trait WithShape<D, C> {
    fn with(device: D, array: C) -> Self;
}

impl<'a, T, D, const N: usize> WithShape<&'a D, [T; N]> for Buffer<'a, T, D, Dim1<N>>
where
    T: Number, // using Number here, because T could be an array type
    D: Alloc<'a, T, Dim1<N>>,
{
    fn with(device: &'a D, array: [T; N]) -> Self {
        Buffer {
            ptr: device.with_array(array),
            device: Some(device),
            node: Default::default(),
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
            ptr: device.with_array(*array),
            device: Some(device),
            node: Default::default(),
        }
    }
}

impl<'a, T, D, const B: usize, const A: usize> WithShape<&'a D, [[T; A]; B]> for Buffer<'a, T, D, Dim2<B, A>>
where
    T: Number,
    D: Alloc<'a, T, Dim2<B, A>>,
{
    fn with(device: &'a D, array: [[T; A]; B]) -> Self {
        Buffer {
            ptr: device.with_array(array),
            device: Some(device),
            node: Default::default(),
        }
    }
}

impl<'a, T, D, const B: usize, const A: usize> WithShape<&'a D, &[[T; A]; B]> for Buffer<'a, T, D, Dim2<B, A>>
where
    T: Number,
    D: Alloc<'a, T, Dim2<B, A>>,
{
    fn with(device: &'a D, array: &[[T; A]; B]) -> Self {
        Buffer {
            ptr: device.with_array(*array),
            device: Some(device),
            node: Default::default(),
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

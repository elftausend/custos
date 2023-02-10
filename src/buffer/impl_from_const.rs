use crate::{shape::Shape, Alloc, Buffer, Dim1};

pub trait WithShape<D, C> {
    fn with(device: D, array: C) -> Self;
}

impl<'a, T, D, const N: usize> WithShape<&'a D, [T; N]> for Buffer<'a, T, D, Dim1<N>>
where
    T: Clone,
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
    T: Copy,
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

impl<'a, T, D, S: Shape> WithShape<&'a D, ()> for Buffer<'a, T, D, S>
where
    D: Alloc<'a, T, S>,
{
    fn with(device: &'a D, _: ()) -> Self {
        Buffer::new(device, S::LEN)
    }
}

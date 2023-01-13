use crate::{shape::Shape, Alloc, Buffer, Dim2};

pub trait WithShape<D, C> {
    fn with(device: D, array: C) -> Self;
}

impl<'a, T, D, S: Shape, const N: usize> WithShape<&'a D, [T; N]> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S>,
{
    fn with(device: &'a D, array: [T; N]) -> Self {
        Buffer {
            ptr: device.with_array(array),
            device: Some(device),
            node: Default::default(),
        }
    }
}

impl<'a, T, D, S: Shape, const N: usize> WithShape<&'a D, &[T; N]> for Buffer<'a, T, D, S>
where
    T: Copy,
    D: Alloc<'a, T, S>,
{
    fn with(device: &'a D, array: &[T; N]) -> Self {
        Buffer {
            ptr: device.with_array(*array),
            device: Some(device),
            node: Default::default(),
        }
    }
}

impl<'a, T, D, const B: usize, const A: usize> WithShape<&'a D, ()> for Buffer<'a, T, D, Dim2<B, A>>
where
    T: Copy,
    D: Alloc<'a, T, Dim2<B, A>>,
{
    fn with(device: &'a D, _array: ()) -> Self {
        Buffer::new(device, Dim2::<B, A>::LEN)
    }
}

/*impl<'a, T, D, S: Shape> WithShape<&'a D, ()> for Buffer<'a, T, D, S>
where
    D: Alloc<'a, T, S>,
{
    fn with(device: &'a D, _: ()) -> Self {
        Buffer::new(device, S::LEN)
    }
}*/

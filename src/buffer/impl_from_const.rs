use crate::{shape::Shape, Alloc, BufFlag, Buffer};

pub trait WithConst<D, C> {
    fn with(device: D, array: C) -> Self;
}

impl<'a, T, D, S: Shape, const N: usize> WithConst<&'a D, [T; N]> for Buffer<'a, T, D, S>
where
    T: Clone,
    D: Alloc<'a, T, S>,
{
    fn with(device: &'a D, array: [T; N]) -> Self {
        Buffer {
            ptr: device.with_array(array),
            len: N,
            device: Some(device),
            flag: BufFlag::None,
            node: Default::default(),
        }
    }
}

impl<'a, T, D, S: Shape, const N: usize> WithConst<&'a D, &[T; N]> for Buffer<'a, T, D, S>
where
    T: Copy,
    D: Alloc<'a, T, S>,
{
    fn with(device: &'a D, array: &[T; N]) -> Self {
        Buffer {
            ptr: device.with_array(*array),
            len: N,
            device: Some(device),
            flag: BufFlag::None,
            node: Default::default(),
        }
    }
}

impl<'a, T, D, S: Shape> WithConst<&'a D, ()> for Buffer<'a, T, D, S>
where
    D: Alloc<'a, T, S>,
{
    fn with(device: &'a D, _: ()) -> Self {
        Buffer::new(device, S::LEN)
    }
}

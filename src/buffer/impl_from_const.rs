use crate::{Alloc, BufFlag, Buffer};

pub trait WithConst<D, C> {
    fn with(device: D, array: C) -> Self;
}

impl<'a, T, D, const N: usize> WithConst<&'a D, [T; N]> for Buffer<'a, T, D, N>
where
    T: Clone,
    D: Alloc<'a, T, N>,
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

impl<'a, T, D, const N: usize> WithConst<&'a D, &[T; N]> for Buffer<'a, T, D, N>
where
    T: Copy,
    D: Alloc<'a, T, N>,
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

impl<'a, T, D, const N: usize> WithConst<&'a D, ()> for Buffer<'a, T, D, N>
where
    D: Alloc<'a, T, N>,
{
    fn with(device: &'a D, _: ()) -> Self {
        Buffer::new(device, N)
    }
}

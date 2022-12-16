use crate::{Alloc, BufFlag, Buffer};

/*impl<'a, T, D, const N: usize> From<(&D, [T; N])> for Buffer<'a, T, D, N>
where
    D: Alloc<'a, T, N>,
    T: Clone
{
    fn from(array: (&D, [T; N])) -> Self {
        Buffer {
            ptr: array.0.with_array(array.1),
            len: N,
            device: Some(&array.0),
            flag: BufFlag::None,
            node: Node::default(),
        }
    }
}*/

// from const size trait -> for all const N: usize buf impls

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

/*

impl<'a, T: Copy, const N: usize> From<(D, &[T; N])> for Buffer<'a, T, D, N> {
    fn from(array: (D, &[T; N])) -> Self {
        Buffer {
            ptr: DArray::new(*array.1),
            len: N,
            device: Some(&D),
            flag: BufFlag::None,
            node: Node::default(),
        }
    }
}

impl<'a, T: Copy, const N: usize> From<(&D, &[T; N])> for Buffer<'a, T, D, N> {
    fn from(array: (&D, &[T; N])) -> Self {
        Buffer {
            ptr: DArray::new(*array.1),
            len: N,
            device: Some(&D),
            flag: BufFlag::None,
            node: Node::default(),
        }
    }
}*/

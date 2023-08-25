use super::stack_device::Stack;
use crate::{
    shape::{Dim1, Dim2},
    Buffer, StackArray,
};

/*impl<'a, T, const N: usize> From<[T; N]> for Buffer<'a, T, Stack, N> {
    fn from(array: [T; N]) -> Self {
        Buffer {
            data: StackArray::new(array),
            device: Some(&Stack::new()),
            node: Node::default(),
        }
    }
}

impl<'a, T: Copy, const N: usize> From<&[T; N]> for Buffer<'a, T, Stack, N> {
    fn from(array: &[T; N]) -> Self {
        Buffer {
            data: StackArray::new(*array),
            device: Some(&Stack::new()),
            node: Node::default(),
        }
    }
}*/

impl<'a, T, const N: usize> From<(&'a Stack, [T; N])> for Buffer<'a, T, Stack, Dim1<N>> {
    fn from((dev, array): (&'a Stack, [T; N])) -> Self {
        Buffer {
            data: StackArray::from_array(array),
            device: Some(dev),
        }
    }
}

impl<'a, T: Copy + Default, const A: usize, const B: usize, const N: usize>
    From<(&'a Stack, [T; N])> for Buffer<'a, T, Stack, Dim2<A, B>>
{
    fn from((dev, array): (&'a Stack, [T; N])) -> Self {
        let mut arr = StackArray::new();
        arr.copy_from_slice(&array);
        Buffer {
            data: arr,
            device: Some(dev),
        }
    }
}

impl<'a, T: Copy, const N: usize> From<(&'a Stack, &[T; N])> for Buffer<'a, T, Stack, Dim1<N>> {
    fn from((dev, array): (&'a Stack, &[T; N])) -> Self {
        Buffer {
            data: StackArray::from_array(*array),
            device: Some(dev),
        }
    }
}

impl<'a, T: Copy + Default, const N: usize, const A: usize, const B: usize>
    From<(&'a Stack, &[T; N])> for Buffer<'a, T, Stack, Dim2<A, B>>
{
    fn from((dev, array): (&'a Stack, &[T; N])) -> Self {
        let mut arr = StackArray::new();
        arr.copy_from_slice(array);
        Buffer {
            // TODO: is this correct
            // ident: Some(Ident::new_bumped(arr.len())),
            data: arr,
            device: Some(dev),
        }
    }
}

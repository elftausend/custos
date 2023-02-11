use super::stack_device::Stack;
use crate::{
    shape::{Dim1, Dim2},
    Buffer, Ident, StackArray
};

/*impl<'a, T, const N: usize> From<[T; N]> for Buffer<'a, T, Stack, N> {
    fn from(array: [T; N]) -> Self {
        Buffer {
            ptr: StackArray::new(array),
            device: Some(&Stack),
            node: Node::default(),
        }
    }
}

impl<'a, T: Copy, const N: usize> From<&[T; N]> for Buffer<'a, T, Stack, N> {
    fn from(array: &[T; N]) -> Self {
        Buffer {
            ptr: StackArray::new(*array),
            device: Some(&Stack),
            node: Node::default(),
        }
    }
}*/

impl<'a, T, const N: usize> From<(Stack, [T; N])> for Buffer<'a, T, Stack, Dim1<N>> {
    fn from((_, array): (Stack, [T; N])) -> Self {
        Buffer {
            ident: Ident::new_bumped(array.len()),
            ptr: StackArray::from_array(array),
            device: Some(&Stack),
        }
    }
}

impl<'a, T, const N: usize> From<(&Stack, [T; N])> for Buffer<'a, T, Stack, Dim1<N>> {
    fn from((_, array): (&Stack, [T; N])) -> Self {
        Buffer {
            ident: Ident::new_bumped(array.len()),
            ptr: StackArray::from_array(array),
            device: Some(&Stack),
        }
    }
}

impl<'a, T: Copy + Default, const A: usize, const B: usize, const N: usize> From<(&Stack, [T; N])>
    for Buffer<'a, T, Stack, Dim2<A, B>>
{
    fn from((_, array): (&Stack, [T; N])) -> Self {
        let mut arr = StackArray::new();
        arr.copy_from_slice(&array);
        Buffer {
            ident: Ident::new_bumped(arr.len()),
            ptr: arr,
            device: Some(&Stack),
        }
    }
}

impl<'a, T: Copy, const N: usize> From<(Stack, &[T; N])> for Buffer<'a, T, Stack, Dim1<N>> {
    fn from((_, array): (Stack, &[T; N])) -> Self {
        Buffer {
            ident: Ident::new_bumped(array.len()),
            ptr: StackArray::from_array(*array),
            device: Some(&Stack),
        }
    }
}

impl<'a, T: Copy, const N: usize> From<(&Stack, &[T; N])> for Buffer<'a, T, Stack, Dim1<N>> {
    fn from((_, array): (&Stack, &[T; N])) -> Self {
        Buffer {
            ident: Ident::new_bumped(array.len()),
            ptr: StackArray::from_array(*array),
            device: Some(&Stack),
        }
    }
}

impl<'a, T: Copy + Default, const N: usize, const A: usize, const B: usize> From<(&Stack, &[T; N])>
    for Buffer<'a, T, Stack, Dim2<A, B>>
{
    fn from((_, array): (&Stack, &[T; N])) -> Self {
        let mut arr = StackArray::new();
        arr.copy_from_slice(array);
        Buffer {
            ident: Ident::new_bumped(arr.len()),
            ptr: arr,
            device: Some(&Stack),
        }
    }
}

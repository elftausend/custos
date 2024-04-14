use super::stack_device::Stack;
use crate::{
    shape::{Dim1, Dim2},
    Buffer, StackArray,
};

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
            data: arr,
            device: Some(dev),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, Stack};

    #[test]
    fn test_from_1d_array_stack() {
        let dev = Stack::new();
        let _buf = dev.with_shape([[1, 2, 3]]);
        // let dev.buffer([1, 2, 3, 4]);
    }
}

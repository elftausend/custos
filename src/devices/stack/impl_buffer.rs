use super::stack_device::Stack;
use crate::{Buffer, IsConstDim, StackArray};

impl<'a, T: Copy + Default, S: IsConstDim, const N: usize> From<(&'a Stack, [T; N])>
    for Buffer<'a, T, Stack, S>
{
    fn from((dev, array): (&'a Stack, [T; N])) -> Self {
        let mut data = StackArray::new();
        data.copy_from_slice(&array);
        Buffer {
            data,
            device: Some(dev),
        }
    }
}

impl<'a, T: Copy + Default, S: IsConstDim, const N: usize> From<(&'a Stack, &[T; N])>
    for Buffer<'a, T, Stack, S>
{
    fn from((dev, array): (&'a Stack, &[T; N])) -> Self {
        let mut data = StackArray::new();
        data.copy_from_slice(array);
        Buffer {
            data,
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

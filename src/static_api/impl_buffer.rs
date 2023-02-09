use crate::Buffer;

use super::static_cpu;

impl<'a, T: Clone> From<&[T]> for Buffer<'a, T> {
    fn from(slice: &[T]) -> Self {
        let device = static_cpu();
        Buffer::from((device, slice))
    }
}

impl<'a, T: Clone, const N: usize> From<&[T; N]> for Buffer<'a, T> {
    fn from(slice: &[T; N]) -> Self {
        let device = static_cpu();
        Buffer::from((device, slice))
    }
}

impl<'a, T: Clone, const N: usize> From<[T; N]> for Buffer<'a, T> {
    fn from(slice: [T; N]) -> Self {
        let device = static_cpu();
        Buffer::from((device, slice))
    }
}

impl<'a, T: Clone> From<Vec<T>> for Buffer<'a, T> {
    fn from(data: Vec<T>) -> Self {
        let device = static_cpu();
        Buffer::from((device, data))
    }
}

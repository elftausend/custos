use crate::{Buffer, Unit};

use super::static_cpu;

impl<'a, T: Unit + Clone> From<&[T]> for Buffer<'a, T> {
    fn from(slice: &[T]) -> Self {
        let device = static_cpu();
        Buffer::from((device, slice))
    }
}

impl<'a, T: Unit + Clone, const N: usize> From<&[T; N]> for Buffer<'a, T> {
    fn from(slice: &[T; N]) -> Self {
        let device = static_cpu();
        Buffer::from((device, slice))
    }
}

impl<'a, T: Unit + Clone, const N: usize> From<[T; N]> for Buffer<'a, T> {
    fn from(slice: [T; N]) -> Self {
        let device = static_cpu();
        Buffer::from((device, slice))
    }
}

impl<'a, T: Unit + Clone> From<Vec<T>> for Buffer<'a, T> {
    fn from(data: Vec<T>) -> Self {
        let device = static_cpu();
        Buffer::from((device, data))
    }
}

use crate::Unit;

use super::{static_cpu, CpuBuffer};

impl<'a, T: Unit + Clone> From<&[T]> for CpuBuffer<'a, T> {
    fn from(slice: &[T]) -> Self {
        let device = static_cpu();
        CpuBuffer::from((device, slice))
    }
}

impl<'a, T: Unit + Clone, const N: usize> From<&[T; N]> for CpuBuffer<'a, T> {
    fn from(slice: &[T; N]) -> Self {
        let device = static_cpu();
        CpuBuffer::from((device, slice))
    }
}

impl<'a, T: Unit + Clone, const N: usize> From<[T; N]> for CpuBuffer<'a, T> {
    fn from(slice: [T; N]) -> Self {
        let device = static_cpu();
        CpuBuffer::from((device, slice))
    }
}

impl<'a, T: Unit + Clone> From<Vec<T>> for CpuBuffer<'a, T> {
    fn from(data: Vec<T>) -> Self {
        let device = static_cpu();
        CpuBuffer::from((device, data))
    }
}

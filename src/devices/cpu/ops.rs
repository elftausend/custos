use core::ops::{AddAssign, RangeBounds, Index};

use crate::{
    ApplyFunction, Buffer, ClearBuf, Device, Eval, MainMemory, Read, Resolve, Shape, ToVal,
    UnaryGrad, WriteBuf, CPU, CopySlice,
};

impl<T, D: MainMemory, S: Shape> Read<T, D, S> for CPU {
    type Read<'a> = &'a [T] where T: 'a, D: 'a, S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, D, S>) -> Self::Read<'a> {
        buf.as_slice()
    }

    #[inline]
    fn read_to_vec<'a>(&self, buf: &Buffer<T, D, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        buf.to_vec()
    }
}

impl<T: Default, D: MainMemory, S: Shape> ClearBuf<T, S, D> for CPU {
    fn clear(&self, buf: &mut Buffer<T, D, S>) {
        for value in buf {
            *value = T::default();
        }
    }
}

impl<T: Copy, D: MainMemory, S: Shape> WriteBuf<T, S, D> for CPU {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, D, S>, data: &[T]) {
        buf.copy_from_slice(data)
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, D, S>, src: &Buffer<T, D, S>) {
        self.write(dst, src)
    }
}

impl<T: Copy, R: RangeBounds<usize>, D: MainMemory> CopySlice<T, R, D> for CPU
where
    [T]: Index<R, Output = [T]>,
{
    fn copy_slice(&self, buf: &Buffer<T, D>, range: R) -> Buffer<T, Self> {
        let slice = &buf.as_slice()[range];
        let mut copied = Buffer::new(self, slice.len());
        self.write(&mut copied, slice);
        copied
    }
}


use custos_macro::impl_stack;

#[cfg(feature = "stack")]
use crate::Stack;

#[impl_stack]
impl<T, D, S> ApplyFunction<T, S, D> for CPU
where
    T: Copy + Default + ToVal,
    D: crate::MainMemory,
    S: Shape,
{
    fn apply_fn<F>(&self, buf: &Buffer<T, D, S>, f: impl Fn(Resolve<T>) -> F) -> Buffer<T, Self, S>
    where
        F: Eval<T>,
    {
        let mut out = self.retrieve::<T, S>(buf.len());

        for (value, x) in out.iter_mut().zip(buf.iter()) {
            *value = f((*x).to_val()).eval()
        }

        out
    }
}

#[impl_stack]
impl<T, D, S> UnaryGrad<T, S, D> for CPU
where
    T: AddAssign + Copy + std::ops::Mul<Output = T>,
    S: Shape,
    D: MainMemory,
{
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F,
    ) where
        F: Eval<T> + ToString,
    {
        for ((lhs, lhs_grad), out) in lhs.iter().zip(lhs_grad.iter_mut()).zip(out.iter()) {
            *lhs_grad += *out * lhs_grad_fn((*lhs).to_val()).eval();
        }
    }
}

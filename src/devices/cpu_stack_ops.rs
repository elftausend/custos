#[cfg(any(feature = "cpu", feature = "stack"))]
use core::ops::AddAssign;

//#[cfg(any(feature = "cpu", feature = "stack"))]
use custos_macro::impl_stack;

use crate::MayToCLSource;
#[cfg(any(feature = "cpu", feature = "stack"))]
use crate::{ApplyFunction, Buffer, Device, Eval, MainMemory, Resolve, Shape, ToVal, UnaryGrad};

#[cfg(feature = "cpu")]
use crate::CPU;

#[cfg(feature = "stack")]
use crate::Stack;

#[impl_stack]
impl<T, D, S> ApplyFunction<T, S, D> for CPU
where
    T: Copy + Default,
    D: crate::MainMemory,
    S: Shape,
{
    fn apply_fn<F>(&self, buf: &Buffer<T, D, S>, f: impl Fn(Resolve<T>) -> F) -> Buffer<T, Self, S>
    where
        F: Eval<T> + MayToCLSource,
    {
        let mut out = self.retrieve::<T, S>(buf.len(), buf);

        for (value, x) in out.iter_mut().zip(buf.iter()) {
            *value = f((*x).to_val()).eval(T::default())
        }

        out
    }
}

#[impl_stack]
impl<T, D, S> UnaryGrad<T, S, D> for CPU
where
    T: AddAssign + Copy + std::ops::Mul<Output = T> + Default,
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
        F: Eval<T> + MayToCLSource,
    {
        for ((lhs, lhs_grad), out) in lhs.iter().zip(lhs_grad.iter_mut()).zip(out.iter()) {
            *lhs_grad += *out * lhs_grad_fn((*lhs).to_val()).eval(T::default());
        }
    }
}

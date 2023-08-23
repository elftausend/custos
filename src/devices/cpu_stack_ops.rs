#[cfg(any(feature = "cpu", feature = "stack"))]
use core::ops::AddAssign;

//#[cfg(any(feature = "cpu", feature = "stack"))]
use custos_macro::impl_stack;

#[cfg(any(feature = "cpu", feature = "stack"))]
use crate::{ApplyFunction, Buffer, Eval, MainMemory, Resolve, Shape, ToVal, UnaryGrad};
use crate::{MayToCLSource, OnDropBuffer, Retrieve, Retriever};

#[cfg(feature = "cpu")]
use crate::CPU;

#[cfg(feature = "stack")]
use crate::Stack;

#[inline]
fn apply_fn_slice<T, O>(x: &[T], out: &mut [T], f: impl Fn(crate::Resolve<T>) -> O)
where
    T: Copy,
    O: Eval<T>,
{
    for (x, out) in x.iter().zip(out.iter_mut()) {
        *out = f((*x).to_val()).eval();
    }
}

#[impl_stack]
impl<T, D, S> ApplyFunction<T, S, D> for CPU
where
    T: Copy + Default + ToVal + 'static,
    D: crate::MainMemory,
    S: Shape,
{
    #[inline]
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, D, S>,
        out: &mut Buffer<T, D, S>,
        f: impl Fn(Resolve<T>) -> F,
    ) where
        F: Eval<T> + MayToCLSource,
    {
        apply_fn_slice(buf, out, f);
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
        F: Eval<T> + MayToCLSource,
    {
        for ((lhs, lhs_grad), out) in lhs.iter().zip(lhs_grad.iter_mut()).zip(out.iter()) {
            *lhs_grad += *out * lhs_grad_fn((*lhs).to_val()).eval();
        }
    }
}

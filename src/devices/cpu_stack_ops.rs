use core::ops::AddAssign;
use core::ops::Mul;

use crate::{Eval, ToVal};

#[inline]
pub fn apply_fn_slice<T, O>(x: &[T], out: &mut [T], f: impl Fn(crate::Resolve<T>) -> O)
where
    T: Copy,
    O: Eval<T>,
{
    for (x, out) in x.iter().zip(out.iter_mut()) {
        *out = f((*x).to_val()).eval();
    }
}

#[inline]
pub fn add_unary_grad<T, O>(
    lhs: &[T],
    out: &[T],
    lhs_grad: &mut [T],
    lhs_grad_fn: impl Fn(crate::Resolve<T>) -> O,
) where
    T: Copy + AddAssign + Mul<Output = T>,
    O: Eval<T>,
{
    for ((lhs, lhs_grad), out) in lhs.iter().zip(lhs_grad.iter_mut()).zip(out.iter()) {
        *lhs_grad += *out * lhs_grad_fn((*lhs).to_val()).eval();
    }
}

#[inline]
pub fn clear_slice<T: Default>(input: &mut [T]) {
    for value in input {
        *value = T::default();
    }
}

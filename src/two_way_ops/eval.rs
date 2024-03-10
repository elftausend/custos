use crate::{prelude::Numeric, Combiner};

/// Evaluates a combined (via [`Combiner`]) math operations chain to a value.
pub trait Eval<T>: 'static {
    /// Evaluates a combined (via [`Combiner`]) math operations chain to a value.
    /// # Example
    /// ```
    /// use custos::{Eval, Combiner};
    ///
    /// let x = 1.5f32.add(2.5).mul(3.5).eval();
    ///
    /// assert_eq!(x, 14.);
    /// ```
    fn eval(&self) -> T;
}

impl<T: Copy + 'static> Eval<T> for T {
    #[inline]
    fn eval(&self) -> T {
        *self
    }
}

impl<T: Numeric> Combiner for T {}

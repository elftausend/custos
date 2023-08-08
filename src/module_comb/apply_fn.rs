use crate::{Eval, MayToCLSource, Resolve, Shape};

use super::{Buffer, Device};

/// Applies a function to a buffer and returns a new buffer.
pub trait ApplyFunction<T, S: Shape = (), D: Device = Self>: Device {
    /// Applies a function to a buffer and returns a new buffer.
    /// # Example
    #[cfg_attr(all(feature = "cpu", feature = "macro"), doc = "```")]
    #[cfg_attr(not(all(feature = "cpu", feature = "macro")), doc = "```ignore")]
    /// use custos::{CPU, Buffer, ApplyFunction, Combiner};
    ///
    /// let device = CPU::new();
    /// let a = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    ///
    /// let out = device.apply_fn(&a, |x| x.mul(2.));
    /// assert_eq!(&*out, &[2., 4., 6., 6., 4., 2.,]);
    /// ```
    fn apply_fn<F>(&self, buf: &Buffer<T, D, S>, f: impl Fn(Resolve<T>) -> F) -> Buffer<T, Self, S>
    where
        F: Eval<T> + MayToCLSource;
}

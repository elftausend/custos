use crate::{Alloc, Buffer, Device, Eval, HasId, MayTapeActions, MayToCLSource, Resolve, Shape};

/// Applies a function to a buffer and returns a new buffer.
pub trait ApplyFunction<T, S: Shape = (), D: Device = Self>: Device {
    /// Applies a function to a buffer and returns a new buffer.
    /// # Example
    #[cfg_attr(all(feature = "cpu", feature = "macro"), doc = "```")]
    #[cfg_attr(not(all(feature = "cpu", feature = "macro")), doc = "```ignore")]
    /// use custos::{CPU, Buffer, ApplyFunction, Combiner, Base};
    ///
    /// let device = CPU::<Base>::new();
    /// let a = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    ///
    /// let out = device.apply_fn(&a, |x| x.mul(2.));
    /// assert_eq!(&*out, &[2., 4., 6., 6., 4., 2.,]);
    /// ```
    #[track_caller]
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>) -> F + Copy,
    ) -> Buffer<T, Self, S>
    where
        F: Eval<T> + MayToCLSource;
}

pub(crate) trait ApplyFunctionLazyTest<T, S: Shape = (), D: Device = Self>: Device {
    #[track_caller]
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>) -> F + Copy,
    ) -> Buffer<T, Self, S>
    where
        F: Eval<T> + MayToCLSource;
}

/// Writes the unary gradient (with chainrule) to the lhs_grad buffer.
pub trait UnaryGrad<T, S: Shape = (), D: Device = Self>: Device {
    /// Write the unary gradient to the lhs_grad buffer.
    /// # Example
    #[cfg_attr(all(feature = "cpu", feature = "macro"), doc = "```")]
    #[cfg_attr(not(all(feature = "cpu", feature = "macro")), doc = "```ignore")]
    /// use custos::{CPU, Buffer, UnaryGrad, Combiner, Base, ToVal};
    ///
    /// let device = CPU::<Base>::new();
    ///
    /// let a = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    /// let out_grad = Buffer::from((&device, [1.; 6]));
    ///
    /// let mut lhs_grad = Buffer::from((&device, [0.; 6]));
    ///
    /// device.add_unary_grad(&a, &mut lhs_grad, &out_grad, |x| 2f64.to_val());
    ///
    /// assert_eq!(&*lhs_grad, &[2.; 6]);
    ///
    /// ```
    #[track_caller]
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        out_grad: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F,
    ) where
        F: Eval<T> + MayToCLSource;
}

/// Applies the forward function of a new/cached [`Buffer`] and returns it.
/// If the `autograd` feature is enabled, the gradient function is also calculated via the grad function.
pub trait UnaryElementWiseMayGrad<T, D: Device, S: Shape>: Device {
    /// Applies the forward function of a new/cached [`Buffer`] and returns it.
    /// If the `autograd` feature is enabled, the gradient function is also calculated via the grad function.
    /// # Example
    #[cfg_attr(
        all(feature = "autograd", feature = "cpu", feature = "macro"),
        doc = "```"
    )]
    #[cfg_attr(
        not(all(feature = "autograd", feature = "cpu", feature = "macro")),
        doc = "```ignore"
    )]
    /// use custos::{CPU, Buffer, UnaryElementWiseMayGrad, Combiner, Base, Autograd, ToVal};
    ///
    /// let device = CPU::<Autograd<Base>>::new();
    ///
    /// let buf = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    /// let out = device.unary_ew(&buf, |x| x.mul(2.), |x| 2f64.to_val());
    ///
    /// assert_eq!(&*out, &[2., 4., 6., 6., 4., 2.,]);
    ///
    /// out.backward();
    /// assert_eq!(&**buf.grad(), &[2.; 6]);
    /// ```
    #[track_caller]
    fn unary_ew<FO, GO>(
        &self,
        buf: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>) -> FO + Copy,
        grad_fn: fn(Resolve<T>) -> GO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + MayToCLSource,
        GO: Eval<T> + MayToCLSource + 'static;
}

impl<T, D, S> UnaryElementWiseMayGrad<T, D, S> for D
where
    T: 'static,
    D: ApplyFunction<T, S, D> + UnaryGrad<T, S, D> + MayTapeActions,
    D: Alloc<T> + 'static,
    S: Shape,
{
    #[inline(always)]
    fn unary_ew<FO, GO>(
        &self,
        buf: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>) -> FO + Copy,
        _grad_fn: fn(Resolve<T>) -> GO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + MayToCLSource,
        GO: Eval<T> + MayToCLSource + 'static,
    {
        let out = self.apply_fn(buf, forward_fn);

        #[cfg(feature = "autograd")]
        {
            let ids = (buf.id(), out.id());
            self.add_grad_fn(move |grads| {
                let (lhs, lhs_grad, out_grad) = grads.get_double::<T, S, S, D>(ids);
                lhs.device()
                    .add_unary_grad(lhs, lhs_grad, out_grad, _grad_fn);
            });
        }

        out
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    #[cfg(feature = "macro")]
    #[test]
    fn test_unary_elementwise() {
        use crate::{Base, Combiner, Device, UnaryElementWiseMayGrad, CPU};

        let device = CPU::<Base>::new();
        let buf = device.buffer([1., 2., 3., 4.]);
        let out = device.unary_ew(&buf, |x| x.sin(), |x| x.cos());

        assert_eq!(
            &*out,
            [
                0.8414709848078965,
                0.9092974268256817,
                0.1411200080598672,
                -0.7568024953079282
            ]
        );
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_may_grad() {
        use crate::{Autograd, Base, Combiner, Device, UnaryElementWiseMayGrad, CPU};

        let device = CPU::<Autograd<Base>>::new();
        let buf = device.buffer([1., 2., 3., 4.]);
        let out = device.unary_ew(&buf, |x| x.sin(), |x| x.cos());

        assert_eq!(
            &*out,
            [
                0.8414709848078965,
                0.9092974268256817,
                0.1411200080598672,
                -0.7568024953079282
            ]
        );
        out.backward();
        assert_eq!(
            &**buf.grad(),
            [
                0.5403023058681398,
                -0.4161468365471424,
                -0.9899924966004454,
                -0.6536436208636119
            ]
        );
    }
}

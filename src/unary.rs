use crate::{Shape, Buffer, Eval, Device, Resolve, Alloc, MayTapeReturn};

/// Applies a function to a buffer and returns a new buffer.
pub trait ApplyFunction<T, S: Shape = (), D: Device = Self>: Device {
    /// Applies a function to a buffer and returns a new buffer.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
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
        F: Eval<T> + ToString;
}

/// Writes the unary gradient (with chainrule) to the lhs_grad buffer.
pub trait UnaryGrad<T, S: Shape = (), D: Device = Self>: Device {
    /// Write the unary gradient to the lhs_grad buffer.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, UnaryGrad, Combiner};
    /// 
    /// let device = CPU::new();
    /// 
    /// let a = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    /// let out_grad = Buffer::from((&device, [1.; 6]));
    /// 
    /// let mut lhs_grad = Buffer::from((&device, [0.; 6]));
    /// 
    /// device.add_unary_grad(&a, &mut lhs_grad, &out_grad, |x| 2.);
    /// 
    /// assert_eq!(&*lhs_grad, &[2.; 6]);
    /// 
    /// ```
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        out_grad: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F,
    ) where
        F: Eval<T> + ToString;
}

/// Applies the forward function of a new/cached [`Buffer`] and returns it.
/// If the `autograd` feature is enabled, the gradient function is also calculated via the grad function.
pub trait UnaryElementWiseMayGrad<T, D: Device, S: Shape>: Device {
    /// Applies the forward function of a new/cached [`Buffer`] and returns it.
    /// If the `autograd` feature is enabled, the gradient function is also calculated via the grad function.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, UnaryElementWiseMayGrad, Combiner};
    /// 
    /// let device = CPU::new();
    /// 
    /// let buf = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    /// let out = device.unary_ew(&buf, |x| x.mul(2.), |x| 2.);
    /// 
    /// assert_eq!(&*out, &[2., 4., 6., 6., 4., 2.,]);
    /// 
    /// out.backward();
    /// assert_eq!(&**buf.grad(), &[2.; 6]);
    /// ```
    fn unary_ew<FO, GO>(
        &self,
        buf: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>) -> FO,
        grad_fn: fn(Resolve<T>) -> GO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + ToString,
        GO: Eval<T> + ToString + 'static;
}

impl<T, D, S> UnaryElementWiseMayGrad<T, D, S> for D
where
    T: 'static,
    D: ApplyFunction<T, S, D> + UnaryGrad<T, S, D> + MayTapeReturn,
    D: for<'b> Alloc<'b, T, S>,
    S: Shape,
{
    #[inline(always)]
    fn unary_ew<FO, GO>(
        &self,
        buf: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>) -> FO,
        _grad_fn: fn(Resolve<T>) -> GO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + ToString,
        GO: Eval<T> + ToString + 'static,
    {
        let out = self.apply_fn(buf, forward_fn);

        #[cfg(feature = "autograd")]
        {
            let ids = (buf.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, lhs_grad, out_grad) = grads.get_double::<T, S, S>(device, ids);
                device.add_unary_grad(&lhs, lhs_grad, out_grad, _grad_fn);
            });
        }

        out
    }
}
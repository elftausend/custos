use crate::{
    AddGradFn, AddOperation, Alloc, AsNoId, Buffer, Device, Eval, HasId, MayTapeActions,
    MayToCLSource, Resolve, Shape, TwoWay, Unit, ZeroGrad,
};

/// Applies a function to a buffer and returns a new buffer.
pub trait ApplyFunction<T: Unit, S: Shape = (), D: Device = Self>: Device {
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
    /// assert_eq!(&**out, &[2., 4., 6., 6., 4., 2.,]);
    /// ```
    fn apply_fn<F>(
        &self,
        // buf: &D::Data<T, S>,
        buf: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>) -> F + Copy + 'static,
    ) -> Buffer<T, Self, S>
    where
        F: TwoWay<T> + 'static;
}

/// Writes the unary gradient (with chainrule) to the lhs_grad buffer.
pub trait UnaryGrad<T: Unit, S: Shape = (), D: Device = Self>: Device {
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
    /// assert_eq!(&**lhs_grad, &[2.; 6]);
    ///
    /// ```
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        out_grad: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F + Copy + 'static,
    ) where
        F: Eval<T> + MayToCLSource;
}

/// Applies the forward function of a new/cached [`Buffer`] and returns it.
/// If the `autograd` feature is enabled, the gradient function is also calculated via the grad function.
pub trait UnaryElementWiseMayGrad<T: Unit, D: Device, S: Shape>: Device {
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
    /// let buf = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,])).require_grad();
    /// let out = device.unary_ew(&buf, |x| x.mul(2.), |x| 2f64.to_val());
    ///
    /// assert_eq!(&**out, &[2., 4., 6., 6., 4., 2.,]);
    ///
    /// out.backward();
    /// assert_eq!(buf.grad().as_slice(), &[2.; 6]);
    /// ```
    fn unary_ew<FO, GO>(
        &self,
        buf: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>) -> FO + Copy + 'static,
        grad_fn: fn(Resolve<T>) -> GO,
    ) -> Buffer<T, Self, S>
    where
        FO: TwoWay<T>,
        GO: Eval<T> + MayToCLSource + 'static;
}

impl<'a, T, D, S> UnaryElementWiseMayGrad<T, D, S> for D
where
    T: Unit + 'static,
    D: AddGradFn + ApplyFunction<T, S, D> + UnaryGrad<T, S, D> + MayTapeActions<'a> + AddOperation + 'a,
    // D::Data<T, S>: crate::ShallowCopy,
    D: Alloc<T> + ZeroGrad<T> + 'static,
    S: Shape,
{
    #[inline(always)]
    fn unary_ew<FO, GO>(
        &self,
        buf: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>) -> FO + Copy + 'static,
        _grad_fn: fn(Resolve<T>) -> GO,
    ) -> Buffer<T, Self, S>
    where
        FO: TwoWay<T>,
        GO: Eval<T> + MayToCLSource + 'static,
    {
        let out = self.apply_fn(buf, forward_fn);

        self.add_grad_fn((buf, &out, _grad_fn.no_id()), |(buf, out, grad_fn)| {
            if !buf.requires_grad() {
                return Ok(());
            }
            // lazy execution is already disabled during backward pass
            buf.device().eagerly(|| {
                buf.device()
                    .add_unary_grad(buf, buf.grad_mut(), out.grad(), **grad_fn);
            });
            Ok(())
        });

        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (buf.id(), out.id());
        //     self.add_grad_fn(move |grads| {
        //         let (lhs, lhs_grad, out_grad) = grads.get_double::<T, S, S, D>(ids);
        //         lhs.device()
        //             .add_unary_grad(lhs, lhs_grad, out_grad, _grad_fn);
        //     });
        // }

        out
    }
}

#[cfg(test)]
mod tests {
    use crate::tests_helper::roughly_eq_slices;

    #[cfg(feature = "cpu")]
    #[cfg(feature = "macro")]
    #[test]
    fn test_unary_elementwise() {
        use crate::{Base, Combiner, Device, UnaryElementWiseMayGrad, CPU};

        let device = CPU::<Base>::new();
        let buf = device.buffer([1., 2., 3., 4.]);
        let out = device.unary_ew(&buf, |x| x.sin(), |x| x.cos());

        roughly_eq_slices(
            &**out,
            &[
                0.8414709848078965,
                0.9092974268256817,
                0.1411200080598672,
                -0.7568024953079282,
            ],
        );
    }

    #[cfg(feature = "autograd")]
    fn test_unary_autograd<'a, D>(device: &'a D)
    where
        D::Data<f32, ()>: crate::ShallowCopy,
        D: 'static
            + crate::WriteBuf<f32>
            + crate::Read<f32>
            + crate::TapeActions<'a>
            + crate::HasAutograd
            + crate::UnaryElementWiseMayGrad<f32, D, ()>
            + crate::Alloc<f32>
            + crate::CachedBuffers
            + crate::AddOperation
            + crate::ZeroGrad<f32>
            + crate::OnNewBuffer<f32, D, ()>,
    {
        use crate::Combiner;

        let buf = device.buffer([1., 2., 3., 4.]).require_grad();
        let out = device.unary_ew(&buf, |x| x.sin(), |x| x.cos());

        roughly_eq_slices(
            &out.read_to_vec(),
            &[
                0.8414709848078965,
                0.9092974268256817,
                0.1411200080598672,
                -0.7568024953079282,
            ],
        );

        out.backward();
        roughly_eq_slices(
            &buf.grad().read_to_vec(),
            &[
                0.5403023058681398,
                -0.4161468365471424,
                -0.9899924966004454,
                -0.6536436208636119,
            ],
        );
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_grad() {
        use crate::{Autograd, Base, CPU};

        let device = CPU::<Autograd<Base>>::new();
        test_unary_autograd(&device)
    }

    #[cfg(feature = "opencl")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_grad_cl() {
        use crate::{Autograd, Base, OpenCL};

        let device = OpenCL::<Autograd<Base>>::new(0).unwrap();
        test_unary_autograd(&device);
    }

    #[cfg(feature = "cuda")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_grad_cu() {
        use crate::{Autograd, Base, CUDA};

        let device = CUDA::<Autograd<Base>>::new(0).unwrap();
        test_unary_autograd(&device);
    }

    #[cfg(feature = "vulkan")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_grad_vk() {
        use crate::{Autograd, Base, Vulkan};

        let device = Vulkan::<Autograd<Base>>::new(0).unwrap();
        test_unary_autograd(&device);
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_may_grad_multiple_times() {
        use crate::{Autograd, Base, Cached, Combiner, Device, UnaryElementWiseMayGrad, CPU};

        let device = CPU::<Autograd<Cached<Base>>>::new();
        let buf = device.buffer([1., 2., 3., 4.]).require_grad();

        for _ in 0..10 {
            let out = device.unary_ew(&buf, |x| x.sin(), |x| x.cos());
            roughly_eq_slices(
                out.as_slice(),
                &[
                    0.8414709848078965,
                    0.9092974268256817,
                    0.1411200080598672,
                    -0.7568024953079282,
                ],
            );

            out.backward();
            assert_eq!(
                buf.grad().as_slice(),
                [
                    0.5403023058681398,
                    -0.4161468365471424,
                    -0.9899924966004454,
                    -0.6536436208636119
                ]
            );

            buf.grad_mut().clear();
        }
    }

    macro_rules! run_several_times {
        ($device:ident, $buf:ident, $out:ident) => {
            for i in 1..10 {
                unsafe { $device.run() }.unwrap();
                roughly_eq_slices(
                    $out.replace().as_slice(),
                    &[
                        0.8414709848078965,
                        0.9092974268256817,
                        0.1411200080598672,
                        -0.7568024953079282,
                    ],
                );
                $out.replace().backward();
                roughly_eq_slices(
                    $buf.replace().grad().as_slice(),
                    &[
                        0.5403023058681398 * i as f64,
                        -0.4161468365471424 * i as f64,
                        -0.9899924966004454 * i as f64,
                        -0.6536436208636119 * i as f64,
                    ],
                );
            }
        };
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_may_grad_multiple_times_lazy() {
        use crate::{Autograd, Base, Combiner, Device, Lazy, Run, UnaryElementWiseMayGrad, CPU};

        let device = CPU::<Autograd<Lazy<Base, f64>>>::new();
        let buf = device.buffer([1., 2., 3., 4.]).require_grad();

        let out = device.unary_ew(&buf, |x| x.sin(), |x| x.cos());
        run_several_times!(device, buf, out);
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_may_grad_multiple_times_lazy_with_lazy_input() {
        use crate::{
            ApplyFunction, Autograd, Base, Combiner, Device, Lazy, Run, UnaryElementWiseMayGrad,
            CPU,
        };

        let device = CPU::<Autograd<Lazy<Base, f64>>>::new();
        let buf = device.buffer([0., 1., 2., 3.]).require_grad();
        let buf1 = device.apply_fn(&buf, |x| x.add(1.));

        let out = device.unary_ew(&buf1, |x| x.sin(), |x| x.cos());
        run_several_times!(device, buf1, out);
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_unary_elementwise_may_grad_multiple_times_lazy_with_lazy_input_exec_last() {
        use crate::{
            ApplyFunction, Autograd, Base, Combiner, Device, ExecNow, Lazy, Run,
            UnaryElementWiseMayGrad, CPU,
        };

        let device = CPU::<Autograd<Lazy<Base, f64>>>::new();
        let buf = device.buffer([0., 1., 2., 3.]).require_grad();
        let buf1 = device.apply_fn(&buf, |x| x.add(1.));

        // this executes the operation above and consumes it
        // this results in apply_fn being only executed once
        device.exec_last_n(&device, 1).unwrap();

        let out = device.unary_ew(&buf1, |x| x.sin(), |x| x.cos());

        run_several_times!(device, buf1, out);
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "autograd")]
    // #[cfg_attr(
    //     miri,
    //     ignore = "location is always different with miri - caching etc does not work"
    // )]
    #[test]
    fn test_unary_elementwise_may_grad_multiple_times_backwards_at_end() {
        use crate::{
            Autograd, Base, Cached, Combiner, Cursor, Device, UnaryElementWiseMayGrad, CPU,
        };

        let device = CPU::<Autograd<Cached<Base>>>::new();
        let buf = device.buffer([1., 2., 3., 4.]).require_grad();

        for i in device.range(0..9) {
            let _out = device.unary_ew(&buf, |x| x.sin(), |x| x.cos());
            if i == 0 {
                _out.grad_mut().write(&[1., 1., 1., 1.]);
            }
        }
        let out = device.unary_ew(&buf, |x| x.sin(), |x| x.cos());
        out.backward();

        roughly_eq_slices(
            buf.grad().as_slice(),
            &[
                0.5403023058681398 * 10.,
                -0.4161468365471424 * 10.,
                -0.9899924966004454 * 10.,
                -0.6536436208636119 * 10.,
            ],
        );
    }
}

mod ops;
mod resolve;

pub use resolve::*;

use self::ops::{Add, Cos, Div, Eq, Exp, GEq, LEq, Mul, Neg, Pow, Sin, Sub, Tan};

/// Evaluates a combined (via [`Combiner`]) math operations chain to a valid OpenCL C (and possibly CUDA) source string.
#[cfg(not(feature = "no-std"))]
pub trait ToCLSource {
    /// Evaluates a combined (via [`Combiner`]) math operations chain to a valid OpenCL C (and possibly CUDA) source string.
    fn to_cl_source(&self) -> String;
}

#[cfg(not(feature = "no-std"))]
impl<N: crate::number::Numeric> ToCLSource for N {
    #[inline]
    fn to_cl_source(&self) -> String {
        self.to_string()
    }
}

/// If the `no-std` feature is disabled, this trait is implemented for all types that implement [`ToCLSource`].
/// In this case, `no-std` is disabled.
#[cfg(not(feature = "no-std"))]
pub trait MayToCLSource: ToCLSource {}
#[cfg(not(feature = "no-std"))]
impl<T: ToCLSource> MayToCLSource for T {}

/// If the `no-std` feature is disabled, this trait is implemented for all types that implement [`ToCLSource`].
/// In this case, `no-std` is enabled and no C source string can be generated.
#[cfg(feature = "no-std")]
pub trait MayToCLSource {}
#[cfg(feature = "no-std")]
impl<T> MayToCLSource for T {}

/// Evaluates a combined (via [`Combiner`]) math operations chain to a value.
pub trait Eval<T> {
    /// Evaluates a combined (via [`Combiner`]) math operations chain to a value.
    /// # Example
    /// ```
    /// use std::ops::{Add, Mul};
    /// use custos::{Eval, Combiner};
    ///
    /// let x = 1.5f32.add(2.5).mul(3.5).eval();
    ///
    /// assert_eq!(x, 14.);
    /// ```
    fn eval(self) -> T;
}

impl<T: Copy> Eval<T> for T {
    #[inline]
    fn eval(self) -> T {
        self
    }
}

/// A trait that allows combining math operations.
/// (Similiar to an Iterator)
pub trait Combiner {
    /// Combines two values into a new one via an addition.
    #[inline]
    fn add<R>(self, rhs: R) -> Add<Self, R>
    where
        Self: Sized,
    {
        Add::new(self, rhs)
    }

    /// Combines two values into a new one via an multiplication.
    #[inline]
    fn mul<R>(self, rhs: R) -> Mul<Self, R>
    where
        Self: Sized,
    {
        Mul::new(self, rhs)
    }

    /// Combines two values into a new one via an subtraction.
    #[inline]
    fn sub<R>(self, rhs: R) -> Sub<Self, R>
    where
        Self: Sized,
    {
        Sub::new(self, rhs)
    }

    /// Combines two values into a new one via an division.
    #[inline]
    fn div<R>(self, rhs: R) -> Div<Self, R>
    where
        Self: Sized,
    {
        Div::new(self, rhs)
    }

    /// Calculates the sine of a value.
    #[inline]
    fn sin(self) -> Sin<Self>
    where
        Self: Sized,
    {
        Sin { comb: self }
    }

    /// Calculates the cosine of a value.
    #[inline]
    fn cos(self) -> Cos<Self>
    where
        Self: Sized,
    {
        Cos { comb: self }
    }

    /// Calculates the tangent of a value.
    #[inline]
    fn tan(self) -> Tan<Self>
    where
        Self: Sized,
    {
        Tan { comb: self }
    }

    /// Combined two values into a new one via exponentiation.
    #[inline]
    fn pow<R>(self, rhs: R) -> Pow<Self, R>
    where
        Self: Sized,
    {
        Pow::new(self, rhs)
    }

    /// Checks if the left value is greater than the right value.
    #[inline]
    fn geq<R>(self, rhs: R) -> GEq<Self, R>
    where
        Self: Sized,
    {
        GEq { comb: self, rhs }
    }

    /// Checks if the left value is less than the right value.
    #[inline]
    fn leq<R>(self, rhs: R) -> LEq<Self, R>
    where
        Self: Sized,
    {
        LEq { comb: self, rhs }
    }

    /// Checks if the left value is equal to the right value.
    #[inline]
    fn eq<R>(self, rhs: R) -> Eq<Self, R>
    where
        Self: Sized,
    {
        Eq { comb: self, rhs }
    }

    /// Negates a value.
    #[inline]
    fn neg(self) -> Neg<Self>
    where
        Self: Sized,
    {
        Neg { comb: self }
    }

    /// Calculates the e^x of a value.
    #[inline]
    fn exp(self) -> Exp<Self>
    where
        Self: Sized,
    {
        Exp { comb: self }
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::Float, Combiner, Eval, Resolve, ToVal};

    #[cfg(not(feature = "no-std"))]
    use crate::{ToCLSource, ToMarker};

    #[test]
    fn test_exp() {
        let f = |x: Resolve<f32>| x.exp();

        let res: f32 = f(1f32.to_val()).eval();
        assert_eq!(res, core::f32::consts::E);

        #[cfg(not(feature = "no-std"))]
        {
            let res = f("x".to_marker()).to_cl_source();
            assert_eq!(res, "exp(x)");
        }
    }

    #[test]
    fn test_neg_tan() {
        let f = |x: Resolve<f32>| x.tan().neg();

        let res: f32 = f(2f32.to_val()).eval();
        roughly_eq_slices(&[res], &[2.1850398]);

        #[cfg(not(feature = "no-std"))]
        {
            let res = f("val".to_marker()).to_cl_source();
            assert_eq!(res, "-(tan(val))")
        }
    }

    #[cfg(not(feature = "no-std"))]
    #[test]
    fn test_pow() {
        let f = |x: Resolve<f32>, y: Resolve<f32>| x.mul(3.).pow(y.add(1.));

        let res = f(3f32.to_val(), 2f32.to_val()).eval();
        assert_eq!(res, 9. * 9. * 9.);

        let res = f("x".to_marker(), "y".to_marker()).to_cl_source();
        assert_eq!(res, "pow((x * 3), (y + 1))")
    }

    #[test]
    fn test_eq() {
        let f = |x: Resolve<i32>, y| x.eq(y);

        let res: i32 = f(3.to_val(), 3.to_val()).eval();
        assert_eq!(res, 1);

        #[cfg(not(feature = "no-std"))]
        {
            let res = f("var_x".to_marker(), "other".to_marker()).to_cl_source();
            assert_eq!(res, "(var_x == other)");
        }
    }

    #[test]
    fn test_geq_relu() {
        let f = |x: Resolve<i32>| x.geq(0).mul(x);

        let res = f(Resolve::with_val(3)).eval();
        assert_eq!(res, 3);

        #[cfg(not(feature = "no-std"))]
        {
            let res = f(Resolve::with_marker("var_x")).to_cl_source();
            assert_eq!(res, "((var_x >= 0) * var_x)");
        }
    }

    #[test]
    fn test_geq() {
        let f = |x: Resolve<i32>| x.geq(4);

        let res = f(Resolve::with_val(3)).eval();
        assert_eq!(res, 0);

        #[cfg(not(feature = "no-std"))]
        {
            let res = f(Resolve::with_marker("var_x")).to_cl_source();
            assert_eq!(res, "(var_x >= 4)");
        }
    }

    #[cfg(not(feature = "no-std"))]
    #[test]
    fn test_eval() {
        let f = |x: Resolve<i32>| x.add(2).add(x.mul(8));

        assert_eq!(f(Resolve::with_val(4)).eval(), 38);
    }

    #[cfg(not(feature = "no-std"))]
    #[test]
    fn test_str_result_two_args() {
        let f = |x: Resolve<f32>, y: Resolve<f32>| x.add(y);

        let r = f("x".to_marker(), "y".to_marker()).to_cl_source();
        assert_eq!("(x + y)", r);
    }

    #[cfg(not(feature = "no-std"))]
    #[test]
    fn test_str_result_two_args2() {
        let f = |x: Resolve<f32>, y: Resolve<f32>| x.add(y).mul(3.6).sub(y);

        let a = f(4f32.to_val(), 3f32.to_val());

        roughly_eq_slices(&[a.eval()], &[22.2]);
        // assert_eq!(a.eval(), 22.2);

        let r = f("x".to_marker(), "y".to_marker()).to_cl_source();
        assert_eq!("(((x + y) * 3.6) - y)", r);
    }

    #[cfg(not(feature = "no-std"))]
    #[test]
    fn test_str_result() {
        let f = |x: Resolve<f32>| x.add(2.).mul(x).add(x.mul(8.)).mul(5.);

        let r = f(Resolve::default()).to_cl_source();
        assert_eq!("((((x + 2) * x) + (x * 8)) * 5)", r);
    }

    pub fn roughly_eq_slices<T: Float>(lhs: &[T], rhs: &[T]) {
        for (a, b) in lhs.iter().zip(rhs) {
            if (*a - *b).abs() >= T::as_generic(0.1) {
                panic!(
                    "Slices 
                    left {lhs:?} 
                    and right {rhs:?} do not equal. 
                    Encountered diffrent value: {a}, {b}"
                )
            }
        }
    }

    #[cfg(all(feature = "cpu", feature = "macro"))]
    #[test]
    fn test_apply_fn_cpu() {
        use crate::{ApplyFunction, Base, Buffer, Combiner, CPU};

        let device = CPU::<Base>::new();

        let buf = Buffer::from((&device, &[3, 3, 4, 5, 3, 2]));

        let buf = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(buf.read(), &[6, 6, 7, 8, 6, 5]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_run_apply_fn_opencl() -> crate::Result<()> {
        use crate::{opencl::chosen_cl_idx, ApplyFunction, Base, Buffer, Combiner, OpenCL};

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;

        let buf = Buffer::from((&device, &[3, 3, 4, 5, 3, 2]));

        let buf = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(buf.read(), &[6, 6, 7, 8, 6, 5]);

        Ok(())
    }

    #[cfg(all(feature = "cpu", feature = "macro"))]
    #[test]
    fn test_run_apply_fn_cpu_more_complex() {
        use crate::{ApplyFunction, Base, Buffer, CPU};

        let device = CPU::<Base>::new();

        let buf = Buffer::from((&device, &[3., 3., 4., 5., 3., 2.]));

        let buf = device.apply_fn(&buf, |x| x.mul(2.).add(4.).sin().mul(x).add(1.));
        roughly_eq_slices(
            buf.read(),
            &[
                -0.6320633326681093,
                -0.6320633326681093,
                -1.1462916720017398,
                5.953036778474352,
                -0.6320633326681093,
                2.978716493246764,
            ],
        );
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_run_apply_fn_opencl_more_complex() -> crate::Result<()> {
        use crate::{opencl::chosen_cl_idx, ApplyFunction, Base, Buffer, OpenCL};

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;

        let buf = Buffer::from((&device, &[3., 3., 4., 5., 3., 2.]));

        let buf = device.apply_fn(&buf, |x| x.mul(2.).add(4.).sin().mul(x).add(1.));
        roughly_eq_slices(
            &buf.read(),
            &[
                -0.6320633326681093,
                -0.6320633326681093,
                -1.1462916720017398,
                5.953036778474352,
                -0.6320633326681093,
                2.978716493246764,
            ],
        );

        Ok(())
    }
}

mod ops;
mod resolve;

pub use resolve::*;

use self::ops::{Add, Cos, Div, Eq, GEq, LEq, Mul, Neg, Pow, Sin, Sub, Tan};

pub trait Eval<T> {
    fn eval(self) -> T;
}

impl<T: Copy> Eval<T> for T {
    #[inline]
    fn eval(self) -> T {
        self
    }
}

pub trait Combiner {
    #[inline]
    fn add<R>(self, rhs: R) -> Add<Self, R>
    where
        Self: Sized,
    {
        Add::new(self, rhs)
    }

    #[inline]
    fn mul<R>(self, rhs: R) -> Mul<Self, R>
    where
        Self: Sized,
    {
        Mul::new(self, rhs)
    }

    #[inline]
    fn sub<R>(self, rhs: R) -> Sub<Self, R>
    where
        Self: Sized,
    {
        Sub::new(self, rhs)
    }

    #[inline]
    fn div<R>(self, rhs: R) -> Div<Self, R>
    where
        Self: Sized,
    {
        Div::new(self, rhs)
    }

    #[inline]
    fn sin(self) -> Sin<Self>
    where
        Self: Sized,
    {
        Sin { comb: self }
    }

    #[inline]
    fn cos(self) -> Cos<Self>
    where
        Self: Sized,
    {
        Cos { comb: self }
    }

    #[inline]
    fn tan(self) -> Tan<Self>
    where
        Self: Sized,
    {
        Tan { comb: self }
    }

    #[inline]
    fn pow<R>(self, rhs: R) -> Pow<Self, R>
    where
        Self: Sized,
    {
        Pow::new(self, rhs)
    }

    #[inline]
    fn geq<R>(self, rhs: R) -> GEq<Self, R>
    where
        Self: Sized,
    {
        GEq { comb: self, rhs }
    }

    #[inline]
    fn leq<R>(self, rhs: R) -> LEq<Self, R>
    where
        Self: Sized,
    {
        LEq { comb: self, rhs }
    }

    #[inline]
    fn eq<R>(self, rhs: R) -> Eq<Self, R>
    where
        Self: Sized,
    {
        Eq { comb: self, rhs }
    }

    #[inline]
    fn neg(self) -> Neg<Self>
    where
        Self: Sized,
    {
        Neg { comb: self }
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::Float, Combiner, Eval, Resolve, ToMarker, ToVal};

    #[test]
    fn test_neg_tan() {
        let f = |x: Resolve<f32>| x.tan().neg();

        let res: f32 = f(2f32.to_val()).eval();
        roughly_eq_slices(&[res], &[2.1850398]);

        let res = f("val".to_marker()).to_string();
        assert_eq!(res, "-(tan(val))")
    }

    #[test]
    fn test_pow() {
        let f = |x: Resolve<f32>, y: Resolve<f32>| x.mul(3.).pow(y.add(1.));

        let res = f(3f32.to_val(), 2f32.to_val()).eval();
        assert_eq!(res, 9. * 9. * 9.);

        let res = f("x".to_marker(), "y".to_marker()).to_string();
        assert_eq!(res, "pow((x * 3), (y + 1))")
    }

    #[test]
    fn test_eq() {
        let f = |x: Resolve<i32>, y| x.eq(y);

        let res: i32 = f(3.to_val(), 3.to_val()).eval();
        assert_eq!(res, 1);

        let res = f("var_x".to_marker(), "other".to_marker()).to_string();
        assert_eq!(res, "(var_x == other)");
    }

    #[test]
    fn test_geq_relu() {
        let f = |x: Resolve<i32>| x.geq(0).mul(x);

        let res = f(Resolve::new(3)).eval();
        assert_eq!(res, 3);

        let res = f(Resolve::with_marker("var_x")).to_string();
        assert_eq!(res, "((var_x >= 0) * var_x)");
    }

    #[test]
    fn test_geq() {
        let f = |x: Resolve<i32>| x.geq(4);

        let res = f(Resolve::new(3)).eval();
        assert_eq!(res, 0);

        let res = f(Resolve::with_marker("var_x")).to_string();
        assert_eq!(res, "(var_x >= 4)");
    }

    #[test]
    fn test_eval() {
        let f = |x: Resolve<i32>| x.add(2).add(x.mul(8));

        assert_eq!(f(Resolve::new(4)).eval(), 38);
    }

    #[test]
    fn test_str_result_two_args() {
        let f = |x: Resolve<f32>, y: Resolve<f32>| x.add(y);

        let r = f("x".to_marker(), "y".to_marker()).to_string();
        assert_eq!("(x + y)", r);
    }

    #[test]
    fn test_str_result_two_args2() {
        let f = |x: Resolve<f32>, y: Resolve<f32>| x.add(y).mul(3.6).sub(y);

        let a = f(4f32.to_val(), 3f32.to_val());

        roughly_eq_slices(&[a.eval()], &[22.2]);

        let r = f("x".to_marker(), "y".to_marker()).to_string();
        assert_eq!("(((x + y) * 3.6) - y)", r);
    }

    #[test]
    fn test_str_result() {
        let f = |x: Resolve<f32>| x.add(2.).mul(x).add(x.mul(8.)).mul(5.);

        let r = f(Resolve::default()).to_string();
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

    #[cfg(feature = "cpu")]
    #[test]
    fn test_apply_fn_cpu() {
        use crate::{ApplyFunction, Buffer, Combiner, CPU};

        let device = CPU::new();

        let buf = Buffer::from((&device, &[3, 3, 4, 5, 3, 2]));

        let buf = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(buf.read(), &[6, 6, 7, 8, 6, 5]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_run_apply_fn_opencl() -> crate::Result<()> {
        use crate::{ApplyFunction, Buffer, Combiner, OpenCL};

        let device = OpenCL::new(0)?;

        let buf = Buffer::from((&device, &[3, 3, 4, 5, 3, 2]));

        let buf = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(buf.read(), &[6, 6, 7, 8, 6, 5]);

        Ok(())
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_run_apply_fn_cpu_more_complex() {
        use crate::{ApplyFunction, Buffer, CPU};

        let device = CPU::new();

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
        use crate::{ApplyFunction, Buffer, OpenCL};

        let device = OpenCL::new(0)?;

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

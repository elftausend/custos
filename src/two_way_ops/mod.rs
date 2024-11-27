mod eval;
mod ops;
mod resolve;
pub use eval::*;

#[cfg(feature = "std")]
mod to_cl_source;
#[cfg(feature = "std")]
pub use to_cl_source::*;

mod combiner;
pub use combiner::*;

#[cfg(feature = "std")]
mod to_wgsl_source;

pub use resolve::*;
#[cfg(feature = "std")]
pub use to_wgsl_source::*;

/// If the `no-std` feature is disabled, this trait is implemented for all types that implement [`ToCLSource`].
/// In this case, `no-std` is enabled and no C source string can be generated.
#[cfg(not(feature = "std"))]
pub trait MayToCLSource {}
#[cfg(not(feature = "std"))]
impl<T> MayToCLSource for T {}

#[cfg(not(feature = "std"))]
pub trait MayToWgslSource {}
#[cfg(not(feature = "std"))]
impl<T> MayToWgslSource for T {}

pub trait TwoWay<T>: Eval<T> + MayToCLSource + MayToWgslSource {}

impl<T, A: Eval<T> + MayToCLSource + MayToWgslSource> TwoWay<T> for A {}

// impl<T> dyn TwoWay<T> + '_ {
//     pub fn eval(&self) -> T
//     where
//         Self: Eval<T>
//     {
//         Eval::<T>::eval(self)
//     }
// }

// impl<T: Eval<F>, F> TwoWay<F> for T {
//     // fn eval(&self) -> F {
//     //     self.eval()
//     // }
// }

#[cfg(test)]
pub mod tests_ex {
    use crate::{Combiner, Eval, Resolve, ToVal};

    #[cfg(not(feature = "std"))]
    use crate::Float;

    #[cfg(feature = "std")]
    use crate::{ToCLSource, ToMarker};

    #[cfg(any(feature = "std", feature = "no-std"))]
    #[test]
    fn test_exp() {
        let f = |x: Resolve<f32>| x.exp();

        let res: f32 = f(1f32.to_val()).eval();
        assert_eq!(res, core::f32::consts::E);

        #[cfg(feature = "std")]
        {
            let res = f("x".to_marker()).to_cl_source();
            assert_eq!(res, "exp(x)");
        }
    }

    #[cfg(any(feature = "std", feature = "no-std"))]
    #[test]
    fn test_neg_tan() {
        use crate::tests_helper::roughly_eq_slices;

        let f = |x: Resolve<f32>| x.tan().neg();

        let res: f32 = f(2f32.to_val()).eval();
        roughly_eq_slices(&[res], &[2.1850398]);

        #[cfg(feature = "std")]
        {
            let res = f("val".to_marker()).to_cl_source();
            assert_eq!(res, "-(tan(val))")
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_pow() {
        let f = |x: Resolve<f32>, y: Resolve<f32>| x.mul(3.).pow(y.add(1.));

        let res = f(3f32.to_val(), 2f32.to_val()).eval();
        assert_eq!(res, 9. * 9. * 9.);

        let res = f("x".to_marker(), "y".to_marker()).to_cl_source();
        assert_eq!(res, "pow((x * 3.0), (y + 1.0))")
    }

    #[test]
    fn test_eq() {
        let f = |x: Resolve<i32>, y| x.eq(y);

        let res: i32 = f(3.to_val(), 3.to_val()).eval();
        assert_eq!(res, 1);

        #[cfg(feature = "std")]
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

        #[cfg(feature = "std")]
        {
            let res = f(Resolve::with_marker("var_x")).to_cl_source();
            assert_eq!(res, "((var_x >= 0) * var_x)");
        }
    }

    #[test]
    fn test_add_3() {
        let f = |x: Resolve<i32>| x.add(3);

        let res = f(Resolve::with_val(3)).eval();
        assert_eq!(res, 6);

        #[cfg(feature = "std")]
        {
            let res = f(Resolve::with_marker("var_x")).to_cl_source();
            assert_eq!(res, "(var_x + 3)");
        }
    }

    #[test]
    fn test_geq() {
        let f = |x: Resolve<i32>| x.geq(4);

        let res = f(Resolve::with_val(3)).eval();
        assert_eq!(res, 0);

        #[cfg(feature = "std")]
        {
            let res = f(Resolve::with_marker("var_x")).to_cl_source();
            assert_eq!(res, "(var_x >= 4)");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_eval() {
        let f = |x: Resolve<i32>| x.add(2).add(x.mul(8));

        assert_eq!(f(Resolve::with_val(4)).eval(), 38);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_str_result_two_args() {
        let f = |x: Resolve<f32>, y: Resolve<f32>| x.add(y);

        let r = f("x".to_marker(), "y".to_marker()).to_cl_source();
        assert_eq!("(x + y)", r);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_str_result_two_args2() {
        use crate::tests_helper::roughly_eq_slices;

        let f = |x: Resolve<f32>, y: Resolve<f32>| x.add(y).mul(3.6).sub(y);

        let a = f(4f32.to_val(), 3f32.to_val());

        roughly_eq_slices(&[a.eval()], &[22.2]);
        // assert_eq!(a.eval(), 22.2);

        let r = f("x".to_marker(), "y".to_marker()).to_cl_source();
        assert_eq!("(((x + y) * 3.6) - y)", r);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_str_result() {
        let f = |x: Resolve<f32>| x.add(2.).mul(x).add(x.mul(8.)).mul(5.);

        let r = f(Resolve::default()).to_cl_source();
        assert_eq!("((((x + 2.0) * x) + (x * 8.0)) * 5.0)", r);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_apply_clip_cpu() {
        use crate::{ApplyFunction, Base, CPU, Device};

        let min = 3.;
        let max = 5.;

        let device = CPU::<Base>::new();

        let x = device.buffer(&[1., 3., 4., 6., 3., 2.]);

        let out = device.apply_fn(&x, move |x| x.max(min).min(max));
        assert_eq!(out.read(), &[3., 3., 4., 5., 3., 3.]);
    }
    #[cfg(feature = "opencl")]
    #[test]
    fn test_apply_clip_cl() {
        use crate::{ApplyFunction, Base, Device, OpenCL, prelude::chosen_cl_idx};

        let min = 3.;
        let max = 5.;

        let device = OpenCL::<Base>::new(chosen_cl_idx()).unwrap();

        let x = device.buffer(&[1., 3., 4., 6., 3., 2.]);

        let out = device.apply_fn(&x, move |x| x.max(min).min(max));
        assert_eq!(out.read(), &[3., 3., 4., 5., 3., 3.]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_apply_fn_cpu() {
        use crate::{ApplyFunction, Base, Buffer, CPU, Combiner};

        let device = CPU::<Base>::new();

        let buf = Buffer::from((&device, &[3, 3, 4, 5, 3, 2]));

        let buf = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(buf.read(), &[6, 6, 7, 8, 6, 5]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_run_apply_fn_opencl() -> crate::Result<()> {
        use crate::{ApplyFunction, Base, Buffer, Combiner, OpenCL, opencl::chosen_cl_idx};

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;

        let buf = Buffer::from((&device, &[3, 3, 4, 5, 3, 2]));

        let buf = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(buf.read(), &[6, 6, 7, 8, 6, 5]);

        Ok(())
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_run_apply_fn_cpu_more_complex() {
        use crate::{ApplyFunction, Base, Buffer, CPU, tests_helper::roughly_eq_slices};

        let device = CPU::<Base>::new();

        let buf = Buffer::from((&device, &[3., 3., 4., 5., 3., 2.]));

        let buf = device.apply_fn(&buf, |x| x.mul(2.).add(4.).sin().mul(x).add(1.));
        roughly_eq_slices(buf.read(), &[
            -0.6320633326681093,
            -0.6320633326681093,
            -1.1462916720017398,
            5.953036778474352,
            -0.6320633326681093,
            2.978716493246764,
        ]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_run_apply_fn_opencl_more_complex() -> crate::Result<()> {
        use crate::{
            ApplyFunction, Base, Buffer, OpenCL, opencl::chosen_cl_idx,
            tests_helper::roughly_eq_slices,
        };

        let device = OpenCL::<Base>::new(chosen_cl_idx())?;

        let buf = Buffer::from((&device, &[3., 3., 4., 5., 3., 2.]));

        let buf = device.apply_fn(&buf, |x| x.mul(2.).add(4.).sin().mul(x).add(1.));
        roughly_eq_slices(&buf.read(), &[
            -0.6320633326681093,
            -0.6320633326681093,
            -1.1462916720017398,
            5.953036778474352,
            -0.6320633326681093,
            2.978716493246764,
        ]);

        Ok(())
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn test_run_apply_fn_vulkan_more_complex() -> crate::Result<()> {
        use crate::{ApplyFunction, Base, Buffer, Vulkan, tests_helper::roughly_eq_slices};

        let device = Vulkan::<Base>::new(0)?;

        let buf = Buffer::from((&device, &[3f32, 3., 4., 5., 3., 2.]));

        let buf = device.apply_fn(&buf, |x| x.mul(2.).add(4.).sin().mul(x).add(1.));
        roughly_eq_slices(&buf.read(), &[
            -0.632_063_3,
            -0.632_063_3,
            -1.146_291_6,
            5.953_037,
            -0.632_063_3,
            2.978_716_6,
        ]);

        Ok(())
    }

    #[cfg(feature = "std")]
    fn test(x: &dyn crate::TwoWay<f32>) {
        x.to_cl_source();
        let _a: f32 = x.eval();
        // x.eval::<f32>();
    }

    #[cfg(feature = "std")]
    fn add_op<T, O: crate::TwoWay<T> + 'static>(
        op: impl Fn(Resolve<T>) -> O + 'static,
        ops: &mut Vec<Box<dyn Fn(Resolve<T>) -> Box<dyn crate::TwoWay<T>>>>,
    ) {
        let dyn_op = move |x: Resolve<T>| {
            let op: Box<dyn crate::TwoWay<T>> = Box::new(op(x));
            op
        };
        ops.push(Box::new(dyn_op));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_storing_of_two_ops() {
        let mut ops = vec![];

        let f = |x: Resolve<f32>| x.ln();
        add_op(f, &mut ops);
        let f = |x: Resolve<f32>| x.cos();
        add_op(f, &mut ops);
        let f = |x: Resolve<f32>| x.sin();
        add_op(f, &mut ops);

        let mut src = String::new();
        let mut val_out = 3.4;
        for op in ops {
            let resolve = Resolve {
                val: val_out,
                marker: "x",
            };

            let out = op(resolve);
            src.push_str(&format!(
                "{marker} = {src};\n",
                marker = resolve.marker,
                src = out.to_cl_source()
            ));
            val_out = out.eval();
            test(&*out);
        }
        println!("src: {src}");
        assert_eq!(val_out, (3.4f32).ln().cos().sin());
        // let out = f(3f32.to_val());
        // test(&out);

        // let y: f32 = f(3f32.to_val()).eval();

        let _src = f("x".to_marker()).to_cl_source();
    }
}

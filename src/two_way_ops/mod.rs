mod ops;
mod resolve;

pub use resolve::*;

use self::ops::{Add, Cos, Mul, Sin};

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
    fn eval<T>(self) -> T
    where
        Self: Eval<T> + Sized,
    {
        Eval::eval(self)
    }

    #[inline]
    fn to_string(&self) -> String
    where
        Self: ToString,
    {
        ToString::to_string(self)
    }

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
}

#[cfg(test)]
#[cfg(feature = "opencl")]
mod tests {

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

    use crate::{
        opencl::{enqueue_kernel, CLBuffer},
        Buffer, CDatatype, Combiner, Device, Eval, MainMemory, OpenCL, Resolve, Shape, CPU, prelude::Float,
    };

    pub trait ApplyFunction<T, S: Shape = (), D: Device = CPU>: Device {
        fn apply_fn<F: Eval<T> + ToString>(
            &self,
            buf: &Buffer<T, D, S>,
            f: impl Fn(Resolve<T>) -> F,
        ) -> Buffer<T, Self, S>;
    }

    impl<T, D, S> ApplyFunction<T, S, D> for CPU
    where
        T: Copy + Default,
        D: MainMemory,
        S: Shape,
    {
        fn apply_fn<F: Eval<T>>(
            &self,
            buf: &Buffer<T, D, S>,
            f: impl Fn(Resolve<T>) -> F,
        ) -> Buffer<T, Self, S> {
            let mut out = self.retrieve::<T, S>(buf.len());

            for (value, x) in out.iter_mut().zip(buf.iter()) {
                *value = f(Resolve::new(*x)).eval()
            }

            out
        }
    }

    impl<T, S> ApplyFunction<T, S, OpenCL> for OpenCL
    where
        T: CDatatype,
        S: Shape,
    {
        fn apply_fn<F: Eval<T> + ToString>(
            &self,
            buf: &Buffer<T, Self, S>,
            f: impl Fn(Resolve<T>) -> F,
        ) -> Buffer<T, Self, S> {
            cl_gen_fn(self, buf, f).unwrap()
        }
    }

    pub fn cl_gen_fn<'a, T, S, F: ToString>(
        device: &'a OpenCL,
        x: &CLBuffer<T, S>,
        f: impl Fn(Resolve<T>) -> F,
    ) -> crate::Result<CLBuffer<'a, T, S>>
    where
        T: CDatatype,
        S: Shape,
    {
        let src = format!(
            "
            __kernel void str_op(__global const {datatype}* lhs, __global {datatype}* out) {{
                size_t id = get_global_id(0);
                {datatype} x = lhs[id];
                out[id] = {operation};
            }}
        ",
            datatype = T::as_c_type_str(),
            operation = f(Resolve::with_marker("x")).to_string()
        );

        let out = device.retrieve::<T, S>(x.len());
        enqueue_kernel(device, &src, [x.len(), 0, 0], None, &[x, &out])?;
        Ok(out)
    }

    #[test]
    fn test_apply_fn_cpu() {
        let device = CPU::new();

        let buf = Buffer::from((&device, &[3, 3, 4, 5, 3, 2]));

        let buf = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(buf.read(), &[6, 6, 7, 8, 6, 5]);
    }

    #[test]
    fn test_run_apply_fn_opencl() -> crate::Result<()> {
        let device = OpenCL::new(0)?;

        let buf = Buffer::from((&device, &[3, 3, 4, 5, 3, 2]));

        let buf = device.apply_fn(&buf, |x| x.add(3));
        assert_eq!(buf.read(), &[6, 6, 7, 8, 6, 5]);

        Ok(())
    
    }

    #[test]
    fn test_run_apply_fn_cpu_more_complex() {
        let device = CPU::new();

        let buf = Buffer::from((&device, &[3., 3., 4., 5., 3., 2.]));

        let buf = device.apply_fn(&buf, |x| x.mul(2.).add(4.).sin().mul(x).add(1.));
        roughly_eq_slices(buf.read(), &[-0.6320633326681093, -0.6320633326681093, -1.1462916720017398, 5.953036778474352, -0.6320633326681093, 2.978716493246764]);
    }

    #[test]
    fn test_run_apply_fn_opencl_more_complex() -> crate::Result<()> {
        let device = OpenCL::new(0)?;

        let buf = Buffer::from((&device, &[3., 3., 4., 5., 3., 2.]));

        let buf = device.apply_fn(&buf, |x| x.mul(2.).add(4.).sin().mul(x).add(1.));
        roughly_eq_slices(&buf.read(), &[-0.6320633326681093, -0.6320633326681093, -1.1462916720017398, 5.953036778474352, -0.6320633326681093, 2.978716493246764]);

        Ok(())
    }
}

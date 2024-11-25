use core::{fmt::Debug, marker::PhantomData};

use crate::Resolve;

pub enum OpHint<T> {
    #[cfg(feature = "std")]
    Unary(std::rc::Rc<dyn Fn(Resolve<T>) -> Box<dyn crate::TwoWay<T>>>),
    None,
    UnaryFused,
    PhantomData(PhantomData<T>),
}

impl<T> Debug for OpHint<T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "std")]
            OpHint::Unary(_) => write!(f, "Unary(...)"),
            OpHint::None => write!(f, "None"),
            OpHint::PhantomData(_) => write!(f, "PhantomData"),
            OpHint::UnaryFused => write!(f, "UnaryFused"),
        }
    }
}

#[cfg(feature = "std")]
pub fn unary<T, O: crate::TwoWay<T> + 'static>(
    op: impl Fn(Resolve<T>) -> O + 'static,
) -> OpHint<T> {
    let dyn_op = move |x: Resolve<T>| {
        let op: Box<dyn crate::TwoWay<T>> = Box::new(op(x));
        op
    };
    OpHint::Unary(std::rc::Rc::new(dyn_op))
}

#[cfg(test)]
mod tests {
    use crate::{op_hint::OpHint, Resolve};

    #[cfg(feature = "cpu")]
    #[cfg(feature = "lazy")]
    #[test]
    fn test_op_hint_update() {
        use crate::{op_hint::OpHint, ApplyFunction, Base, Combiner, Device, Lazy, Resolve, CPU};

        let dev = CPU::<Lazy<Base>>::new();

        let buf = dev.buffer([1., 2., 3., 4., 5.]);
        let out = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&out, |x| x.cos());
        let _out = dev.apply_fn(&out, |x| x.ln());

        let resolve = Resolve {
            val: 7.,
            marker: "x",
        };

        let ops = &dev.modules.graph.borrow().operations;
        let op_hint = &ops[0].op_hint;
        if let OpHint::Unary(op) = op_hint {
            let src = op(resolve).to_cl_source();
            assert_eq!(src, "sin(x)");
        } else {
            panic!()
        }

        let op_hint = &ops[1].op_hint;
        if let OpHint::Unary(op) = op_hint {
            let src = op(resolve).to_cl_source();
            assert_eq!(src, "cos(x)");
        } else {
            panic!()
        }

        let op_hint = &ops[2].op_hint;
        if let OpHint::Unary(op) = op_hint {
            let src = op(resolve).to_cl_source();
            assert_eq!(src, "log(x)");
        } else {
            panic!()
        }
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "lazy")]
    #[test]
    fn test_op_hint_unary_chain_fuse() {
        use crate::{ApplyFunction, Base, Combiner, Device, Lazy, CPU};

        let dev = CPU::<Lazy<Base>>::new();

        let buf = dev.buffer([1., 2., 3., 4., 5.]);
        let out = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&out, |x| x.cos());
        let _out = dev.apply_fn(&out, |x| x.ln());

        let mut out = buf.clone();

        for out in out.iter_mut() {
            for op in &dev.modules.graph.borrow().operations {
                let resolve = Resolve {
                    val: *out,
                    marker: "x",
                };
                if let OpHint::Unary(op) = &op.op_hint {
                    *out = op(resolve).eval();
                }
            }
        }

        for (buf, out) in buf.iter().zip(out.iter()) {
            assert!((*out - buf.sin().cos().ln()).abs() < 0.001);
        }
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[test]
    fn test_op_hint_unary_chain_fuse_graph() {
        use crate::{ApplyFunction, Base, Combiner, Device, ChangePtr, Lazy, Optimize, Run, CPU};

        let dev = CPU::<Graph<Lazy<Base>>>::new();

        let buf = dev.buffer([1., 2., 3., 4., 5.]);
        let out = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&out, |x| x.cos());
        let _out = dev.apply_fn(&out, |x| x.ln());

        dev.optimize_mem_graph(&dev, None).unwrap();
        dev.unary_fusing(&dev, None).unwrap();
        dev.run().unwrap();

        for (buf, out) in buf.iter().zip(_out.replace().iter()) {
            assert!((*out - buf.sin().cos().ln()).abs() < 0.001);
        }
    }

    #[cfg(feature = "opencl")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[test]
    fn test_op_hint_unary_chain_fuse_graph_cl() {
        use crate::{ApplyFunction, Base, Combiner, Device, ChangePtr, Lazy, OpenCL, Optimize, Run};

        let dev = OpenCL::<Graph<Lazy<Base>>>::new(0).unwrap();

        let buf = dev.buffer([1f32, 2., 3., 4., 5.]);
        let out = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&out, |x| x.cos());
        let _out = dev.apply_fn(&out, |x| x.ln());

        dev.optimize_mem_graph(&dev, None).unwrap();
        dev.unary_fusing(&dev, None).unwrap();
        dev.run().unwrap();

        for (buf, out) in buf.read().iter().zip(_out.replace().read().iter()) {
            assert!((*out - buf.sin().cos().ln()).abs() < 0.001);
        }
    }

    #[cfg(feature = "cuda")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[test]
    fn test_op_hint_unary_chain_fuse_graph_cu() {
        use crate::{ApplyFunction, Base, Combiner, Device, ChangePtr, Lazy, Optimize, Run, CUDA};

        let dev = CUDA::<Graph<Lazy<Base>>>::new(0).unwrap();

        let buf = dev.buffer([1f32, 2., 3., 4., 5.]);
        let out = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&out, |x| x.cos());
        let _out = dev.apply_fn(&out, |x| x.ln());

        dev.optimize_mem_graph(&dev, None).unwrap();
        dev.unary_fusing(&dev, None).unwrap();
        let _ = dev.run();

        for (buf, out) in buf.read().iter().zip(_out.replace().read().iter()) {
            assert!((*out - buf.sin().cos().ln()).abs() < 0.001);
        }
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[test]
    fn test_op_hint_unary_chain_fuse_graph_complex() {
        use crate::{ApplyFunction, Base, Combiner, Device, ChangePtr, Lazy, Optimize, Run, CPU};

        let dev = CPU::<Graph<Lazy<Base>>>::new();

        let buf = dev.buffer([1., 2., 3., 4., 5.]);
        let rhs = dev.buffer([8., 2., 3., 4., 5.]);
        let out1 = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&rhs, |x| x.sin());
        let _out2 = dev.apply_fn(&out, |x| x.cos());
        let out1 = dev.apply_fn(&out1, |x| x.abs());
        let _out = dev.apply_fn(&out1, |x| x.ln());

        dev.optimize_mem_graph(&dev, None).unwrap();
        dev.unary_fusing(&dev, None).unwrap();
        dev.run().unwrap();

        for (buf, out) in buf.iter().zip(_out.replace().iter()) {
            assert_eq!(*out, buf.sin().abs().ln());
        }

        for (buf, out) in rhs.iter().zip(_out2.replace().iter()) {
            assert_eq!(*out, buf.sin().cos());
        }
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "lazy")]
    #[ignore = "too long runtime"]
    #[test]
    fn test_op_hint_unary_chain_fuse_manual_perf() {
        use std::time::Instant;

        use crate::{ApplyFunction, Base, Combiner, Device, Lazy, CPU};

        let dev = CPU::<Lazy<Base>>::new();

        let buf = dev.buffer::<_, (), _>(vec![1.; N]);
        let out = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&out, |x| x.cos());
        let _out = dev.apply_fn(&out, |x| x.ln());

        let mut out = buf.clone();

        let start = Instant::now();

        for out in out.iter_mut() {
            for op in &dev.modules.graph.borrow().operations {
                let resolve = Resolve {
                    val: *out,
                    marker: "x",
                };
                if let OpHint::Unary(op) = &op.op_hint {
                    *out = op(resolve).eval();
                } else {
                    panic!()
                }
            }
        }

        println!("cpu automatic fusing: {:?}", start.elapsed());

        let mut should = buf.clone();

        let start = Instant::now();

        for should in should.iter_mut() {
            *should = should.sin().cos().ln();
        }

        println!("cpu manual fusing: {:?}", start.elapsed());

        for (should, out) in should.iter().zip(out.iter()) {
            assert_eq!(out, should);
        }
    }

    const N: usize = 100000000;

    #[cfg(feature = "opencl")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[ignore = "too long runtime"]
    #[test]
    fn test_op_hint_unary_chain_fuse_automatic_perf() {
        use std::time::Instant;

        use crate::{ApplyFunction, Base, Combiner, Device, ChangePtr, Lazy, OpenCL, Optimize, Run};

        // let dev = crate::CPU::<Graph<Lazy<Base>>>::new();
        let dev = OpenCL::<Graph<Lazy<Base>>>::new(0).unwrap();

        let buf = dev.buffer::<_, (), _>(vec![1.; N]);
        let out = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&out, |x| x.cos());
        let out = dev.apply_fn(&out, |x| x.ln());

        let start = Instant::now();
        dev.optimize_mem_graph(&dev, None).unwrap();
        println!("optimize mem graph: {:?}", start.elapsed());
        let start = Instant::now();

        dev.unary_fusing(&dev, None).unwrap();

        println!("unary fusing: {:?}", start.elapsed());

        dev.run().unwrap();

        let start = Instant::now();

        dev.run().unwrap();

        println!("perf automatic fusing: {:?}", start.elapsed());

        for (buf, out) in buf.read().iter().zip(out.replace().read().iter()) {
            assert!((*out - buf.sin().cos().ln()).abs() < 0.001);
        }
    }

    #[cfg(feature = "opencl")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[test]
    fn test_cl_fused_kernel_creation() {
        use crate::{Base, ChangePtr, Lazy, OpenCL};

        let _dev = OpenCL::<Graph<Lazy<Base>>>::new(0).unwrap();
    }
}

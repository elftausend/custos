use core::marker::PhantomData;

use crate::Resolve;

pub enum OpHint<T> {
    #[cfg(feature = "std")]
    Unary(Box<dyn Fn(Resolve<T>) -> Box<dyn crate::TwoWay<T>>>),
    None,
    PhantomData(PhantomData<T>),
}

#[cfg(feature = "std")]
pub fn unary<T, O: crate::TwoWay<T> + 'static>(
    op: impl Fn(Resolve<T>) -> O + 'static,
) -> OpHint<T> {
    let dyn_op = move |x: Resolve<T>| {
        let op: Box<dyn crate::TwoWay<T>> = Box::new(op(x));
        op
    };
    OpHint::Unary(Box::new(dyn_op))
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
            assert_eq!(*out, buf.sin().cos().ln());
        }
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[test]
    fn test_op_hint_unary_chain_fuse_graph() {
        use crate::{ApplyFunction, Base, Combiner, Device, Graph, Lazy, Optimize, CPU};

        let dev = CPU::<Graph<Lazy<Base>>>::new();

        let buf = dev.buffer([1., 2., 3., 4., 5.]);
        let out = dev.apply_fn(&buf, |x| x.sin());
        let out = dev.apply_fn(&out, |x| x.cos());
        let _out = dev.apply_fn(&out, |x| x.ln());

        let cts = dev.modules.graph_trans.borrow().opt_graph.cache_traces();
        println!("{cts:?}");

        let mut out = buf.clone();

        for out in out.iter_mut() {
            for op in &dev.modules.modules.graph.borrow().operations {
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
            assert_eq!(*out, buf.sin().cos().ln());
        }
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "lazy")]
    #[ignore = "too long runtime"]
    #[test]
    fn test_op_hint_unary_chain_fuse_perf() {
        use std::time::Instant;

        use crate::{ApplyFunction, Base, Combiner, Device, Lazy, CPU};

        let dev = CPU::<Lazy<Base>>::new();

        const N: usize = 100000000;

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
}

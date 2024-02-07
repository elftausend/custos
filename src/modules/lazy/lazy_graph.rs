use crate::{
    bounds_to_range, AsAny, Buffers, Device, Parents, UniqueId, UpdateArgs, UpdateArgsDynable,
};
use core::{mem::transmute, ops::RangeBounds};

use super::{
    exec_iter::{exec_op, ExecIter},
    generic_support::BoxedShallowCopy,
};

pub struct LazyGraph<B = Box<dyn BoxedShallowCopy>> {
    pub ids_to_check: Vec<Vec<Option<UniqueId>>>,
    pub ops: Vec<fn(*mut ()) -> crate::Result<()>>,
    pub args: Vec<Box<dyn UpdateArgsDynable<B>>>,
}

impl<B> Default for LazyGraph<B> {
    #[inline]
    fn default() -> Self {
        Self {
            ids_to_check: Vec::default(),
            ops: Vec::default(),
            args: Vec::default(),
        }
    }
}

impl<B: AsAny> LazyGraph<B> {
    #[inline]
    pub fn iter_with<'a>(&'a mut self, buffers: &'a mut Buffers<B>) -> ExecIter<B> {
        ExecIter {
            ids_to_check: self.ids_to_check.iter(),
            ops: self.ops.iter(),
            args: self.args.iter_mut(),
            buffers,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.ids_to_check.clear();
        self.ops.clear();
        self.args.clear();
    }

    pub fn add_operation<Args: Parents<N> + UpdateArgs, const N: usize>(
        &mut self,
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    ) {
        // store ids and test if buffers are still in cache
        self.ids_to_check.push(
            args.maybe_ids()
                .into_iter()
                .map(|id| id.map(|id| *id))
                .collect(),
        );

        let args: Box<dyn UpdateArgsDynable<B>> = Box::new(args);

        self.args.push(unsafe { transmute(args) });
        unsafe { self.ops.push(transmute(op)) };
    }

    pub unsafe fn call_lazily<D: Device + 'static>(
        &mut self,
        outs_unordered: &mut Buffers<B>,
    ) -> crate::Result<()> {
        for args in self.iter_with(outs_unordered) {
            args?;
        }
        // for ((args, op), ids_to_check) in
        //     self.args.iter_mut().zip(&self.ops).zip(&self.ids_to_check)
        // {
        //     args.update_args(&ids_to_check, outs_unordered)?;

        //     let args = &mut **args as *mut _ as *mut ();
        //     op(args)?;
        // }
        Ok(())
    }

    pub unsafe fn call_range<D: Device + 'static>(
        &mut self,
        bounds: impl RangeBounds<usize>,
        outs_unordered: &mut Buffers<B>,
    ) -> crate::Result<()> {
        let range = bounds_to_range(bounds, self.args.len());
        for ((mut args, op), ids_to_check) in self
            .args
            .drain(range.clone())
            .zip(self.ops.drain(range.clone()))
            .zip(self.ids_to_check.drain(range.clone()))
        {
            exec_op(&mut args, &op, &ids_to_check, outs_unordered)?;
            // args.update_args(&ids_to_check, outs_unordered)?;

            // let args = &mut *args as *mut _ as *mut ();
            // op(args)?;
        }

        Ok(())
    }
}

#[cfg(feature = "cpu")]
#[cfg(test)]
mod tests {
    use super::LazyGraph;
    use crate::{
        modules::lazy::{generic_support::BoxedShallowCopy, register_buf_copyable},
        AsNoId, Base, Buffer, Device, HasId, Retriever, CPU,
    };
    use std::collections::HashMap;

    #[test]
    #[should_panic]
    fn test_lazy_op_args_args_out_of_scope() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();
        let mut outs_unordered = HashMap::default();

        let _out_id = {
            let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
            let rhs = device.buffer([1f32, 2., 6., 4., 5.]);
            let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
            unsafe { register_buf_copyable(&mut outs_unordered, &out) };
            // outs_unordered.insert(out.id(), )

            graph.add_operation::<_, 3>((&out, &lhs, &rhs), |args| {
                let (_out, lhs, rhs) = *args;
                assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
                assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);
                Ok(())
            });

            out.id()
        };

        // todo!()
        unsafe { graph.call_lazily::<CPU>(&mut outs_unordered).unwrap() }
    }

    #[test]
    fn test_lazy_op_args() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
        unsafe { register_buf_copyable(&mut outs_unordered, &lhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &rhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &out) };
        // outs_unordered.insert(out.id(), )

        graph.add_operation::<_, 3>((&out, &lhs, &rhs), |args| {
            let (_out, lhs, rhs) = *args;
            assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
            assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);
            Ok(())
        });

        unsafe { graph.call_lazily::<CPU>(&mut outs_unordered).unwrap() }
    }

    #[test]
    fn test_lazy_op_args_no_out_but_use_loop() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let mut out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
        unsafe { register_buf_copyable(&mut outs_unordered, &lhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &rhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &out) };
        // outs_unordered.insert(out.id(), )

        for _ in 0..10 {
            graph.add_operation::<_, 3>((&lhs, &rhs, &mut out), |(lhs, rhs, _out)| {
                // println!("ih");
                assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
                assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);

                // if _out.is_some() {
                //     panic!();
                // }
                Ok(())
            });
        }

        unsafe { graph.call_lazily::<CPU>(&mut outs_unordered).unwrap() }
    }
    #[test]
    fn test_lazy_op_args_no_out_but_use() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
        unsafe { register_buf_copyable(&mut outs_unordered, &lhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &rhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &out) };
        // outs_unordered.insert(out.id(), )

        graph.add_operation::<_, 2>((&lhs, &rhs), |args| {
            let (lhs, rhs) = *args;
            assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
            assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);

            Ok(())
        });

        unsafe { graph.call_lazily::<CPU>(&mut outs_unordered).unwrap() }
    }

    #[test]
    fn test_lazy_op_args_with_ew_fn() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let mut out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
        unsafe { register_buf_copyable(&mut outs_unordered, &lhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &rhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &out) };

        let ew_fn = |x: f32| x + 10.;

        // outs_unordered.insert(out.id(), )

        graph.add_operation::<_, 4>(
            (&mut out, &lhs, &rhs, ew_fn.no_id()),
            |(_out, lhs, rhs, ew_fn)| {
                assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
                assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);

                for (out, lhs) in _out.iter_mut().zip(lhs.iter()) {
                    *out = ew_fn(*lhs);
                }

                Ok(())
            },
        );

        unsafe { graph.call_lazily::<CPU>(&mut outs_unordered).unwrap() }
    }

    #[test]
    fn test_lazy_graph_exec_with_vecs() {
        let mut graph = LazyGraph::<Box<dyn BoxedShallowCopy>>::default();

        {
            let vec = vec![1, 2, 3, 4];
            graph.add_operation::<_, 1>(vec.no_id(), |vec| {
                assert_eq!(vec.as_slice(), &[1, 2, 3, 4]);
                Ok(())
            });
        }
        unsafe { graph.call_lazily::<CPU>(&mut HashMap::default()) }.unwrap();
    }

    #[test]
    fn test_args_ref_updating() {
        let x = 5;
        let y = 3.;
        let mut args = (&x, 10, &y);

        let replace_x = &x;
        args.0 = replace_x;
    }
}

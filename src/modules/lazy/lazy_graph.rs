use crate::{
    bounds_to_range, modules::lazy::exec_iter::ExecIter, op_hint::OpHint, AnyOp, BoxedShallowCopy,
    Buffers, Device, Downcast, Id, Parents,
};
use core::ops::RangeBounds;
use std::collections::HashSet;

pub struct Operation<B, T> {
    pub arg_ids: Vec<Id>,
    pub op: Box<dyn Fn(&[Id], &mut Buffers<B>, &dyn core::any::Any) -> crate::Result<()> + 'static>,
    pub op_hint: OpHint<T>,
}

impl<B, T> Operation<B, T> {
    #[inline]
    pub fn no_op() -> Self {
        Self {
            op: Box::new(|_ids, _buffers, _dev| Ok(())),
            arg_ids: vec![],
            op_hint: OpHint::None,
        }
    }

    #[inline]
    pub fn call<D: Device + 'static>(
        &self,
        buffers: &mut Buffers<B>,
        device: &D,
    ) -> crate::Result<()> {
        (self.op)(&self.arg_ids, buffers, device)
    }
}

pub struct LazyGraph<B = Box<dyn BoxedShallowCopy>, T = ()> {
    pub(crate) operations: Vec<Operation<B, T>>,
}

impl<B, T> Default for LazyGraph<B, T> {
    #[inline]
    fn default() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
}

impl<B: Downcast, T> LazyGraph<B, T> {
    #[inline]
    pub fn iter_with<'b, D: Device>(
        &'b mut self,
        device: &'b D,
        buffers: &'b mut Buffers<B>,
    ) -> ExecIter<'b, B, T, D> {
        ExecIter {
            operations: self.operations.iter(),
            buffers,
            device,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.operations.clear();
    }

    #[inline]
    pub fn ops_count(&self) -> usize {
        self.operations.len()
    }

    pub unsafe fn call_lazily<D: Device + 'static>(
        &mut self,
        device: &D,
        buffers: &mut Buffers<B>,
    ) -> crate::Result<()> {
        for args in self.iter_with(device, buffers) {
            args?;
        }
        Ok(())
    }

    pub unsafe fn call_range<D: Device + 'static>(
        &mut self,
        device: &D,
        bounds: impl RangeBounds<usize>,
        buffers: &mut Buffers<B>,
    ) -> crate::Result<()> {
        let range = bounds_to_range(bounds, self.operations.len());
        for op in self.operations.drain(range) {
            op.call(buffers, device)?;
        }
        Ok(())
    }

    pub fn convert_to_operation<Args: Parents<N> + AnyOp, const N: usize>(
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) -> Operation<B, T> {
        const { assert!(N > 0, "Size of parents must be greater than 0") };

        let mut seen_ids = HashSet::new();

        // store ids and test if buffers are still in cache
        let arg_ids = args
            .maybe_ids()
            .into_iter()
            .flat_map(|id| {
                // return error / none
                let id = id.expect("every parent must have an id");
                if seen_ids.contains(&id.id) {
                    panic!("each parent (id) must be unique")
                }
                seen_ids.insert(id.id);

                Some(id)
            })
            .collect::<Vec<_>>();

        if arg_ids.len() != N {
            panic!()
        }

        let op: Box<dyn Fn(&[Id], &mut Buffers<B>, &dyn core::any::Any) -> crate::Result<()>> =
            Args::replication_fn::<B>(op);

        Operation {
            arg_ids,
            op,
            op_hint: OpHint::None,
        }
    }

    pub fn add_operation<Args: Parents<N> + AnyOp, const N: usize>(
        &mut self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) {
        let operation = Self::convert_to_operation(args, op);
        self.operations.push(operation)
    }
}

#[cfg(feature = "cpu")]
#[cfg(test)]
mod tests {
    use crate::{
        register_buf_any, register_buf_copyable, Base, Buffer, Device, HasId, LazyGraph,
        Retriever, CPU,
    };
    use std::collections::HashMap;
    #[cfg(feature = "autograd")]
    #[test]
    fn test_autograd_lazy_op() {
        // static mut DEVICE: Option<&'static CPU<crate::Autograd<Base>>> = None;
        thread_local! {
            static DEVICE2: std::cell::Cell<Option<&'static CPU<crate::CachedModule<Base, CPU>>>> = std::cell::Cell::new(None);
        };
        {
            let device = CPU::<crate::Autograd<'_, Base>>::new();
            let _lhs = device.buffer([1f32, 2., 3., 4., 5.]);
            let _rhs = device.buffer([1f32, 2., 6., 4., 5.]);
            // let mut buffers = HashMap::default();
            // unsafe { register_buf_copyable(&mut buffers, &lhs) };
            // unsafe { register_buf_copyable(&mut buffers, &rhs) };
            // let tape: &mut LazyGraph2 = &mut unsafe {device.modules.tape_mut()}.unwrap().lazy_graph;
            // tape.add_operation((&lhs, &rhs), |(lhs, rhs)| {
            //     lhs.grad();
            //     Ok(())
            // });
            // tape.call_lazily(&device, &mut buffers).unwrap();

            let device = CPU::<crate::Autograd<Base>>::new();
            let mut buffers = HashMap::default();
            let mut graph: LazyGraph<Box<dyn core::any::Any>> = LazyGraph::default();
            let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
            let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

            unsafe { register_buf_any(&mut buffers, &lhs) };
            unsafe { register_buf_any(&mut buffers, &rhs) };

            // buffers.insert(*lhs.id(), &lhs);
            // register_buf_any(cache, buf)
            graph.add_operation::<_, 1>(&lhs, |args| {
                println!("args: {args:?}");

                // DEVICE2.set(args.device);
                // unsafe {
                //     DEVICE = args.device;
                // }
                // std::mem::replace(&mut device, Some(3)) ;
                Ok(())
            });

            graph.add_operation::<_, 2>((&lhs, &rhs), |args| {
                println!("args: {args:?}");
                Ok(())
            });
            unsafe { graph.call_lazily(&device, &mut buffers).unwrap() };
        };
        // let x = DEVICE2.get().unwrap();
        // println!("{:?}", x.modules.cache.borrow().nodes);
        // unsafe { DEVICE.unwrap() };

        // graph.ex

        // graph.add_operation2::<_, 2>((&lhs, &rhs), |args| {
        //     Ok(())
        // });
    }

    #[test]
    #[should_panic]
    fn test_lazy_op_args_args_out_of_scope() {
        let device = CPU::<Base>::new();
        let mut graph: LazyGraph = LazyGraph::default();
        let mut outs_unordered = HashMap::default();

        let _out_id = {
            let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
            let rhs = device.buffer([1f32, 2., 6., 4., 5.]);
            let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs)).unwrap();
            unsafe { register_buf_copyable(&mut outs_unordered, &out) };
            // outs_unordered.insert(out.id(), )

            graph.add_operation::<_, 3>((&out, &lhs, &rhs), |args| {
                let (_out, lhs, rhs) = args;
                assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
                assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);
                Ok(())
            });

            out.id()
        };

        // todo!()
        unsafe { graph.call_lazily(&device, &mut outs_unordered).unwrap() }
    }

    #[test]
    fn test_lazy_op_args() {
        let device = CPU::<Base>::new();
        let mut graph: LazyGraph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs)).unwrap();
        unsafe { register_buf_copyable(&mut outs_unordered, &lhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &rhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &out) };
        // outs_unordered.insert(out.id(), )

        graph.add_operation::<_, 3>((&out, &lhs, &rhs), |args| {
            let (_out, lhs, rhs) = args;
            assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
            assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);
            Ok(())
        });

        unsafe { graph.call_lazily(&device, &mut outs_unordered).unwrap() }
    }

    #[test]
    fn test_lazy_op_args_no_out_but_use_loop() {
        let device = CPU::<Base>::new();

        let mut graph: LazyGraph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let mut out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs)).unwrap();
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

        unsafe { graph.call_lazily(&device, &mut outs_unordered).unwrap() }
    }
    #[test]
    fn test_lazy_op_args_no_out_but_use() {
        let device = CPU::<Base>::new();

        let mut graph: LazyGraph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs)).unwrap();
        unsafe { register_buf_copyable(&mut outs_unordered, &lhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &rhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &out) };
        // outs_unordered.insert(out.id(), )

        graph.add_operation::<_, 2>((&lhs, &rhs), |args| {
            let (lhs, rhs) = args;
            assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
            assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);

            Ok(())
        });

        unsafe { graph.call_lazily(&device, &mut outs_unordered).unwrap() }
    }

    #[test]
    fn test_lazy_op_args_with_ew_fn() {
        let device = CPU::<Base>::new();

        let mut graph: LazyGraph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let mut out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs)).unwrap();
        unsafe { register_buf_copyable(&mut outs_unordered, &lhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &rhs) };
        unsafe { register_buf_copyable(&mut outs_unordered, &out) };

        let ew_fn = |x: f32| x + 10.;

        // outs_unordered.insert(out.id(), )

        graph.add_operation::<_, 3>((&mut out, &lhs, &rhs), move |(_out, lhs, rhs)| {
            assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
            assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);

            for (out, lhs) in _out.iter_mut().zip(lhs.iter()) {
                *out = ew_fn(*lhs);
            }

            Ok(())
        });

        unsafe { graph.call_lazily(&device, &mut outs_unordered).unwrap() }
    }
}

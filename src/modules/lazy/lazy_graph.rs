use crate::{
    bounds_to_range, modules::lazy::exec_iter::ExecIter2, op_hint::OpHint, AnyBuffer, AsAny,
    BoxedShallowCopy, Buffer, Buffers, Device, Downcast, HasId, Parents, Shape, UniqueId,
    UpdateArgs, UpdateArgsDynable,
};
use core::{any::Any, cell::RefCell, marker::PhantomData, mem::transmute, ops::RangeBounds};
use std::collections::{HashMap, HashSet};

use super::exec_iter::{exec_op, ExecIter};

pub trait ArgsTest {}

pub struct Operation2<'a, B, T> {
    // pub op: fn(*mut ()) -> crate::Result<()>,
    pub op: Box<dyn Fn(&mut Buffers<B>) -> crate::Result<()> + 'a>,
    pub op_hint: OpHint<T>,
}

pub struct Operation<B, T> {
    pub op_hint: OpHint<T>,
    pub arg_ids: Vec<Option<UniqueId>>,
    pub op: fn(*mut ()) -> crate::Result<()>,
    pub args: Box<dyn UpdateArgsDynable<B>>,
}

impl<B: AsAny, T> Operation<B, T> {
    pub fn no_op() -> Self {
        Self {
            op_hint: OpHint::None,
            arg_ids: vec![None],
            op: |_: *mut ()| Ok(()),
            args: Box::new(()),
        }
    }
}

pub struct LazyGraph<B = Box<dyn BoxedShallowCopy>, T = ()> {
    pub operations: Vec<Operation<B, T>>,
}

pub struct LazyGraph2<'a, B = Box<dyn BoxedShallowCopy>, T = ()> {
    operations: Vec<Operation2<'a, B, T>>,
}

impl<'a, B, T> Default for LazyGraph2<'a, B, T> {
    #[inline]
    fn default() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
}

impl<'a, B: Downcast, T> LazyGraph2<'a, B, T> {
    #[inline]
    pub fn iter_with<'b>(
        &'b mut self,
        // device: &'a D,
        buffers: &'b mut Buffers<B>,
    ) -> ExecIter2<'a, 'b, B, T> {
        ExecIter2 {
            operations: self.operations.iter(),
            buffers,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.operations.clear();
    }

    pub fn call_lazily(&mut self, buffers: &mut Buffers<B>) -> crate::Result<()> {
        for args in self.iter_with(buffers) {
            args?;
        }
        Ok(())
    }

    pub fn call_range<D: Device + 'static>(
        &mut self,
        _device: &'a D,
        bounds: impl RangeBounds<usize>,
        buffers: &mut Buffers<B>,
    ) -> crate::Result<()> {
        let range = bounds_to_range(bounds, self.operations.len());
        for op in self.operations.drain(range) {
            (op.op)(buffers)?;
        }
        Ok(())
    }

    pub fn convert_to_operation<Args: Parents<N> + AnyOp, const N: usize>(
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) -> Operation2<'a, B, T> {
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
            //.flat_map(|id| id.map(|id| *id))
            .collect::<Vec<_>>();

        if arg_ids.len() != N {
            panic!()
        }

        let op: Box<dyn Fn(&mut Buffers<B>) -> crate::Result<()>> =
            Args::replication_fn::<B>(arg_ids, op);

        Operation2 {
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

impl<B, T> Default for LazyGraph<B, T> {
    #[inline]
    fn default() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
}

pub struct X {
    // x: Box<dyn AnyOp>,
}

pub trait AnyOp: Sized {
    type Replicated<'a>;
    // fn replication_fn(ids: Vec<crate::Id>, op: impl Fn(Self) -> crate::Result<()>) -> impl Fn(&mut BorrowCache) -> crate::Result<()>;
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn for<'i> Fn(&'i mut Buffers<B>) -> crate::Result<()>>;
}

type BorrowCache = HashMap<UniqueId, Box<dyn Any>>;

pub trait Replicate {
    type Replication<'r>: 'r;
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate
    for &crate::Buffer<'a, T, D, S>
{
    type Replication<'r> = Buffer<'r, T, D, S>;
}
// impl<T, D: Device, S: Shape> From<&Buffer<'_, T, D, S>> for &Buffer<'_, T, D, S> {
//     fn from(value: &Buffer<'_, T, D, S>) -> Self {
//         todo!()
//     }
// }

impl<R: HasId + Replicate> AnyOp for R {
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&mut Buffers<B>) -> crate::Result<()>> {
        let id = ids[0];
        Box::new(move |buffers| {
            let r1 = unsafe {
                &mut *(buffers
                    .get_mut(&*id)
                    .unwrap()
                    .downcast_mut::<R::Replication<'_>>()
                    .unwrap() as *mut _)
            };
            op(r1)
        })
    }
    type Replicated<'a> = &'a mut R::Replication<'a>;
}

impl<R1: HasId + Replicate, R2: HasId + Replicate> AnyOp for (R1, R2) {
    fn replication_fn<B: Downcast>(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&mut Buffers<B>) -> crate::Result<()>> {
        let r1 = ids[0];
        let r2 = ids[1];
        Box::new(move |buffers| {
            let r1 = unsafe {
                &mut *(buffers
                    .get_mut(&*r1)
                    .unwrap()
                    .downcast_mut::<R1::Replication<'_>>()
                    .unwrap() as *mut _)
            };
            let r2 = unsafe {
                &mut *(buffers
                    .get_mut(&*r2)
                    .unwrap()
                    .downcast_mut::<R2::Replication<'_>>()
                    .unwrap() as *mut _)
            };
            op((r1, r2))
        })
    }
    type Replicated<'a> = (&'a mut R1::Replication<'a>, &'a mut R2::Replication<'a>);
}
// impl<R1: HasId + for<'r> Replicate<'r>, R2: HasId + for<'r> Replicate<'r>> AnyOp for (R1, R2) {
//     fn replication_fn(ids: Vec<crate::Id>, op: impl Fn(Self) -> crate::Result<()>) -> impl Fn(&mut BorrowCache) -> crate::Result<()> {
//         let r1 = ids[0];
//         let r2 = ids[1];
//         move |buffers| {
//             op((R1::replicate(buffers, r1)?, R2::replicate(buffers, r2)?))
//         }
//     }
// }

// impl<R1: HasId + Replicate3, R2: HasId + Replicate3> AnyOp for (R1, R2) {
//     fn replication_fn(ids: Vec<crate::Id>, op: impl Fn(Self::Replicated) -> crate::Result<()> + 'static) -> Box<dyn Fn(&mut BorrowCache) -> crate::Result<()>> {
//         Box::new(|_| Ok(()))
//     }

//     type Replicated = (R1::Out, R2::Out);
// }

impl<B: AsAny, T> LazyGraph<B, T> {
    #[inline]
    pub fn iter_with<'a>(&'a mut self, buffers: &'a mut Buffers<B>) -> ExecIter<B, T> {
        ExecIter {
            operations: self.operations.iter_mut(),
            buffers,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.operations.clear();
    }

    pub unsafe fn convert_to_operation<Args: Parents<N> + UpdateArgs, const N: usize>(
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    ) -> Operation<B, T> {
        // store ids and test if buffers are still in cache
        let arg_ids = args
            .maybe_ids()
            .into_iter()
            .map(|id| id.map(|id| *id))
            .collect();

        let args: Box<dyn UpdateArgsDynable<B>> = Box::new(args);

        Operation {
            arg_ids,
            op: transmute(op),
            args: transmute(args),
            op_hint: OpHint::None,
        }
    }

    pub fn add_operation<Args: Parents<N> + UpdateArgs, const N: usize>(
        &mut self,
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    ) {
        let operation = unsafe { Self::convert_to_operation(args, op) };
        self.operations.push(operation)
    }

    pub unsafe fn call_lazily<D: Device + 'static>(
        &mut self,
        outs_unordered: &mut Buffers<B>,
    ) -> crate::Result<()> {
        for args in self.iter_with(outs_unordered) {
            args?;
        }
        Ok(())
    }

    pub unsafe fn call_range<D: Device + 'static>(
        &mut self,
        bounds: impl RangeBounds<usize>,
        outs_unordered: &mut Buffers<B>,
    ) -> crate::Result<()> {
        let range = bounds_to_range(bounds, self.operations.len());
        for mut op in self.operations.drain(range) {
            exec_op(&mut op.args, &op.op, &op.arg_ids, outs_unordered)?;
        }
        Ok(())
    }
}

#[cfg(feature = "cpu")]
#[cfg(test)]
mod tests {
    use super::LazyGraph;
    use crate::{
        register_buf_any, register_buf_copyable, AnyBuffer, AsNoId, Base, BoxedShallowCopy, Buffer,
        CloneBuf, Device, HasId, LazyGraph2, Retriever, ShallowCopy, Shape, TapeActionsLT,
        UniqueId, CPU,
    };
    use core::cell::Cell;
    use std::{
        collections::HashMap,
        sync::{Mutex, RwLock},
    };

    pub(crate) fn register_buf_an_bufsy<'a, T, D, S>(
        cache: &mut HashMap<UniqueId, Box<dyn AnyBuffer + 'a>, impl core::hash::BuildHasher>,
        buf: &Buffer<'a, T, D, S>,
    ) where
        T: crate::Unit + 'static,
        D: Device + crate::IsShapeIndep + 'static + CloneBuf<'a, T, S>,
        D::Data<T, S>: crate::ShallowCopy,
        S: Shape,
    {
        // shallow copy sets flag to AllocFlag::Wrapper

        // let wrapped_data = unsafe { buf.data.shallow() };

        // let buf = Buffer {
        //     data: wrapped_data,
        //     device: buf.device,
        // };
        let buf2 = buf.device().clone_buf(&buf);
        cache.insert(*buf.id(), Box::new(buf2));
    }

    #[test]
    fn test_op2() {
        // static mut DEVICE: Option<&'static CPU<crate::Autograd<Base>>> = None;
        thread_local! {
            static DEVICE2: Cell<Option<&'static CPU<crate::CachedModule<Base, CPU>>>> = Cell::new(None);
        };
        // static DEVICES: std::sync::Mutex<Option<&'static CPU<crate::Autograd<Base>>>> = Default::default();
        {
            let device = CPU::<crate::Autograd<'_, Base>>::new();
            let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
            let rhs = device.buffer([1f32, 2., 6., 4., 5.]);
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
            let mut graph: LazyGraph2<Box<dyn core::any::Any>> = LazyGraph2::default();
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
            graph.call_lazily(&mut buffers).unwrap();
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

        unsafe { graph.call_lazily::<CPU>(&mut outs_unordered).unwrap() }
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

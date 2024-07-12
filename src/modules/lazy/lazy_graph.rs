use crate::{
    bounds_to_range, op_hint::OpHint, AnyBuffer, AsAny, BoxedShallowCopy, Buffer, Buffers, Device,
    HasId, Parents, Shape, UniqueId, UpdateArgs, UpdateArgsDynable,
};
use core::{cell::RefCell, marker::PhantomData, mem::transmute, ops::RangeBounds};
use std::collections::{HashMap, HashSet};

use super::exec_iter::{exec_op, ExecIter};

pub trait ArgsTest {}

pub struct Operation2<'a, 'b> {
    pub arg_ids: Vec<Option<UniqueId>>,
    // pub op: fn(*mut ()) -> crate::Result<()>,
    pub op3: Box<dyn Fn(&mut BorrowCache) -> crate::Result<()> + 'a>,
    pub pd: PhantomData<&'b ()>, //pub op2: Box<dyn Fn(*mut ()) -> crate::Result<()>>, // pub args: Box<dyn UpdateArgsDynable<B>>,
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

pub struct LazyGraph2<'a, 'b, B = Box<dyn BoxedShallowCopy>, T = ()> {
    pub operations: Vec<Operation<B, T>>,
    pub operations2: Vec<Operation2<'a, 'a>>,
    _pd: PhantomData<&'b ()>,
}

impl<'a, 'b, B, T> Default for LazyGraph2<'a, 'b, B, T> {
    #[inline]
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            operations2: Vec::new(),
            _pd: PhantomData,
        }
    }
}

impl<'a, 'c, B, T> LazyGraph2<'a, 'c, B, T> {
    #[inline]
    pub fn iter_with(&self, buffers: &'a RefCell<BorrowCache>) -> ExecIter<B, T> {
        for op in &self.operations2 {
            // let mut buffers = buffers.borrow_mut();
            // (op.op3)(&mut buffers);
        }
        todo!()
    }

    pub fn convert_to_operation2<Args: Parents<N> + AnyOp<'a>, const N: usize>(
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) -> Operation2<'a, 'a> {
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

        let op: Box<dyn Fn(&mut BorrowCache) -> crate::Result<()>> =
            Args::replication_fn(arg_ids, op);

        Operation2 {
            arg_ids: vec![],
            op3: op,
            pd: PhantomData,
        }
    }

    pub fn add_operation2<Args: Parents<N> + AnyOp<'a>, const N: usize>(
        &mut self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) {
        let operation = Self::convert_to_operation2(args, op);
        self.operations2.push(operation)
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

pub trait AnyOp<'r>: Sized {
    type Replicated<'a>;
    // fn replication_fn(ids: Vec<crate::Id>, op: impl Fn(Self) -> crate::Result<()>) -> impl Fn(&mut BorrowCache) -> crate::Result<()>;
    fn replication_fn(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn for<'i> Fn(&'i mut BorrowCache) -> crate::Result<()> + 'r>;
}

type BorrowCache = HashMap<UniqueId, Box<dyn AnyBuffer>>;

pub trait Replicate<'a>: Sized {
    fn replicate(buffers: &'a mut BorrowCache, id: crate::Id) -> crate::Result<Self>;
}

pub trait Replicate3: Sized {
    type Out;
    // type Out<'r, 'g>: ConvertTo<Self> where 'g: 'r, 'r: 't;
    fn replicate<'r>(buffers: &'r mut BorrowCache, id: crate::Id) -> crate::Result<&'r Self::Out>;
}

pub trait Replicate2: Sized {
    type Replication<'r, 'a>
    where
        'a: 'r;
    fn replicate<'r, 'a>(
        buffers: &'r mut BorrowCache,
        id: crate::Id,
    ) -> crate::Result<Self::Replication<'r, 'a>>;
}

impl<'a, 'r, T: 'static, D: Device + 'static, S: crate::Shape> Replicate<'r>
    for &'r mut crate::Buffer<'a, T, D, S>
{
    fn replicate(buffers: &'r mut BorrowCache, id: crate::Id) -> crate::Result<Self> {
        todo!()
        // Ok(unsafe { buffers.get_buf_mut(id) }?)
    }
}

// impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate3<'a> for crate::Buffer<'a, T, D, S> {
//     fn replicate<'r>(buffers: &'r mut BorrowCache, id: crate::Id) -> crate::Result<Self::Out<'r, 'a>> {
//         //todo!()
//         Ok(unsafe { buffers.get_buf(id) }?)
//     }

//     type Out<'r, 'g> = &'r Buffer<'g, T, D, S> where 'g: 'r;
// }

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate3
    for &'a crate::Buffer<'_, T, D, S>
{
    type Out = Buffer<'a, T, D, S>;

    fn replicate<'r>(buffers: &'r mut BorrowCache, id: crate::Id) -> crate::Result<&'r Self::Out> {
        todo!()
        // Ok(unsafe { buffers.get_buf(id) }?)
    }

    // type Out<'r, 'g> = &'r Buffer<'g, T, D, S> where 'g: 'r, 'r: 't;
}

// impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate3 for &crate::Buffer<'a, T, D, S> {
//     fn replicate<'r>(buffers: &'r mut BorrowCache, id: crate::Id) -> crate::Result<&'r Self> {
//         //todo!()
//         Ok(unsafe { buffers.get_buf::<T, D, S>(id) }?)
//     }
// }

impl<'a, 'r, T: 'static, D: Device + 'static, S: crate::Shape> Replicate<'r>
    for &'r crate::Buffer<'a, T, D, S>
{
    fn replicate(buffers: &'r mut BorrowCache, id: crate::Id) -> crate::Result<Self> {
        todo!()
        // Ok(unsafe { buffers.get_buf(id) }?)
    }
}

impl<T: 'static, D: Device + 'static, S: crate::Shape> Replicate2 for &crate::Buffer<'_, T, D, S> {
    fn replicate<'r, 'a>(
        buffers: &'r mut BorrowCache,
        id: crate::Id,
    ) -> crate::Result<Self::Replication<'r, 'a>> {
        todo!()

        // Ok(unsafe { buffers.get_buf(id) }?)
    }

    type Replication<'r, 'a> = &'r crate::Buffer<'a, T, D, S> where 'a: 'r;
}

pub trait Replicate4 {
    type Replication: 'static;
}

impl<'a, T: 'static, D: Device + 'static, S: crate::Shape> Replicate4
    for &crate::Buffer<'a, T, D, S>
{
    type Replication = Buffer<'static, T, D, S>;
}
// impl<T, D: Device, S: Shape> From<&Buffer<'_, T, D, S>> for &Buffer<'_, T, D, S> {
//     fn from(value: &Buffer<'_, T, D, S>) -> Self {
//         todo!()
//     }
// }

impl<'r, R: HasId + Replicate4> AnyOp<'r> for R {
    fn replication_fn(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&mut BorrowCache) -> crate::Result<()>> {
        // Box::new(|_| Ok(()))
        let id = ids[0];
        Box::new(move |buffers| {
            let r1 = buffers
                .get_mut(&*id)
                .unwrap()
                .downcast_mut::<R::Replication>()
                .unwrap();

            op(r1)
        })
    }

    // type Replicated<'a> = ();

    type Replicated<'a> = &'a mut R::Replication;

    // type Replicated<'a> = &'a R::Out where R::Out: 'a;
}

impl<'r, R1: HasId + Replicate2, R2: HasId + Replicate2> AnyOp<'r> for (R1, R2) {
    fn replication_fn(
        ids: Vec<crate::Id>,
        op: impl for<'a> Fn(Self::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> Box<dyn Fn(&mut BorrowCache) -> crate::Result<()>> {
        // Box::new(|_| Ok(()))
        let id1 = ids[0];
        let id2 = ids[1];
        // check for distinct ids!
        Box::new(move |buffers| {
            // op((R1::replicate(buffers, id1)?, R2::replicate(buffers, id2)?))
            // op(R::replicate(buffers, id)?)
            Ok(())
        })
    }

    // type Replicated<'a> = ();

    type Replicated<'a> = (R1::Replication<'r, 'r>, R2::Replication<'r, 'r>);

    // type Replicated<'a> = &'a R::Out where R::Out: 'a;
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
        register_buf_copyable, AsNoId, Base, BoxedShallowCopy, Buffer, Device, HasId, LazyGraph2,
        Retriever, CPU,
    };
    use std::collections::HashMap;

    #[test]
    fn test_op2() {
        let device = CPU::<Base>::new();
        let mut graph: LazyGraph2 = LazyGraph2::default();
        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        graph.add_operation2::<_, 1>(&lhs, |args| Ok(()));

        // graph.add_operation2::<_, 2>((&lhs, &rhs), |args| {
        //     Ok(())
        // });

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

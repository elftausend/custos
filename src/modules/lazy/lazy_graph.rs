use crate::{
    bounds_to_range, AllParents, Buffer, Device, DeviceError, Id, NoHasher, Parents, PtrConv,
    Shape, UniqueId, HashLocation,
};
use core::{any::Any, hash::BuildHasherDefault, mem::transmute, ops::RangeBounds, panic::Location};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct LazyGraph {
    pub ids_to_check: Vec<Vec<UniqueId>>,
    pub ops: Vec<fn(*mut (), *mut ()) -> crate::Result<()>>,
    // if a location was already used -> error
    consumed_locations: HashSet<HashLocation<'static>>,
    // indexmap?
    consumed_locations_order: Vec<HashLocation<'static>>,
    pub args: Vec<Box<dyn AllParents>>,
}

impl LazyGraph {
    // TODO: could use a broader range of Args! (limited to Parents<N>)
    #[track_caller]
    pub fn add_operation<T, D, S, Args: Parents<N>, const N: usize>(
        &mut self,
        args: Args,
        op: fn(&mut Option<&mut Buffer<T, D, S>>, &mut Args) -> crate::Result<()>,
    ) -> crate::Result<()> 
    where
        D: PtrConv,
        S: Shape,
    {
        if self.consumed_locations.contains(&Location::caller().into()) {
            return Err(DeviceError::LocationAlreadyInUse.into())
        }

        // store ids and test if buffers are still in cache
        self.ids_to_check.push(
            args.maybe_ids()
                .into_iter()
                .flatten()
                .map(|id| *id)
                .collect(),
        );

        self.consumed_locations.insert(Location::caller().into());

        // let args = Box::leak(Box::new(args));
        // let args: Box<dyn Any> = unsafe { transmute::<Box<dyn Any + 'static>, _>(Box::new(args)) };

        let args: Box<dyn AllParents> = Box::new(args);

        // self.args.push(args as *mut Args as *mut _);
        self.args.push(unsafe { transmute(args) });
        unsafe { self.ops.push(transmute(op)) };

        Ok(())
    }

    pub unsafe fn call_lazily<D: Device + 'static>(
        &mut self,
        out_buf_order: &[Option<Id>],
        outs_unordered: &mut HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
    ) -> crate::Result<()> {
        for (((args, op), ids_to_check), out_id) in self
            .args
            .iter_mut()
            .zip(&self.ops)
            .zip(&self.ids_to_check)
            .zip(out_buf_order)
        {
            for id_to_check in ids_to_check.iter() {
                outs_unordered
                    .get(id_to_check)
                    .ok_or(DeviceError::InvalidLazyBuf)?;
            }
            let args = &mut **args as *mut _ as *mut ();
            match out_id {
                Some(out_id) => {
                    let mut val = outs_unordered.get_mut(out_id).map(|out| &mut **out);

                    let out = &mut val as *mut _ as *mut ();
                    op(out, args)?;
                }
                None => {
                    let mut val = None::<*mut ()>;
                    let out = &mut val as *mut _ as *mut ();
                    op(out, args)?;
                }
            };
        }
        Ok(())
    }

    pub unsafe fn call_range<D: Device + 'static>(
        &mut self,
        bounds: impl RangeBounds<usize>,
        out_buf_order: &mut Vec<Option<Id>>,
        outs_unordered: &mut HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
    ) -> crate::Result<()> {
        let range = bounds_to_range(bounds, out_buf_order.len());
        for ((((mut args, op), ids_to_check), out_id), location) in self
            .args
            .drain(range.clone())
            .zip(self.ops.drain(range.clone()))
            .zip(self.ids_to_check.drain(range.clone()))
            .zip(out_buf_order.drain(range.clone()))
            .zip(self.consumed_locations_order.drain(range))
        {
            self.consumed_locations.remove(&location);

            for id_to_check in ids_to_check.iter() {
                outs_unordered
                    .get(id_to_check)
                    .ok_or(DeviceError::InvalidLazyBuf)?;
            }

            let args = &mut *args as *mut _ as *mut ();

            match out_id {
                Some(out_id) => {
                    let mut val = outs_unordered.get_mut(&out_id).map(|out| &mut **out);

                    let out = &mut val as *mut _ as *mut ();
                    op(out, args)?;
                }
                None => {
                    let mut val = None::<*mut ()>;
                    let out = &mut val as *mut _ as *mut ();
                    op(out, args)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::LazyGraph;
    use crate::{register_buf, AsNoId, Base, Buffer, Device, HasId, Retriever, CPU};
    use std::collections::HashMap;

    #[test]
    #[should_panic]
    fn test_lazy_op_args_args_out_of_scope() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();
        let mut outs_unordered = HashMap::default();

        let out_id = {
            let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
            let rhs = device.buffer([1f32, 2., 6., 4., 5.]);
            let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
            unsafe { register_buf(&mut outs_unordered, &out) };
            // outs_unordered.insert(out.id(), )

            graph.add_operation::<f32, CPU, (), _, 2>((&lhs, &rhs), |_out, args| {
                let (lhs, rhs) = *args;
                assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
                assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);
                Ok(())
            }).unwrap();

            out.id()
        };

        unsafe {
            graph
                .call_lazily::<CPU>(&[Some(out_id)], &mut outs_unordered)
                .unwrap()
        }
    }

    #[test]
    fn test_lazy_op_args() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
        unsafe { register_buf(&mut outs_unordered, &lhs) };
        unsafe { register_buf(&mut outs_unordered, &rhs) };
        unsafe { register_buf(&mut outs_unordered, &out) };
        // outs_unordered.insert(out.id(), )

        graph.add_operation::<f32, CPU, (), _, 2>((&lhs, &rhs), |_out, args| {
            let (lhs, rhs) = *args;
            assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
            assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);
            Ok(())
        }).unwrap();

        unsafe {
            graph
                .call_lazily::<CPU>(&[Some(out.id())], &mut outs_unordered)
                .unwrap()
        }
    }

    #[test]
    fn test_lazy_op_args_no_out_but_use() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
        unsafe { register_buf(&mut outs_unordered, &lhs) };
        unsafe { register_buf(&mut outs_unordered, &rhs) };
        unsafe { register_buf(&mut outs_unordered, &out) };
        // outs_unordered.insert(out.id(), )

        graph.add_operation::<f32, CPU, (), _, 2>((&lhs, &rhs), |_out, args| {
            let (lhs, rhs) = *args;
            assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
            assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);

            if _out.is_some() {
                panic!();
            }
            Ok(())
        }).unwrap();

        unsafe {
            graph
                .call_lazily::<CPU>(&[None], &mut outs_unordered)
                .unwrap()
        }
    }

    #[test]
    fn test_lazy_op_args_with_ew_fn() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
        unsafe { register_buf(&mut outs_unordered, &lhs) };
        unsafe { register_buf(&mut outs_unordered, &rhs) };
        unsafe { register_buf(&mut outs_unordered, &out) };

        let ew_fn = |x: f32| x + 10.;

        // outs_unordered.insert(out.id(), )

        graph.add_operation::<f32, CPU, (), _, 3>((&lhs, &rhs, ew_fn.no_id()), |_out, args| {
            let (lhs, rhs, ew_fn) = *args;
            assert_eq!(lhs.as_slice(), &[1f32, 2., 3., 4., 5.,]);
            assert_eq!(rhs.as_slice(), &[1f32, 2., 6., 4., 5.,]);

            for (out, lhs) in _out.as_mut().unwrap().iter_mut().zip(lhs.iter()) {
                *out = ew_fn(*lhs);
            }

            Ok(())
        }).unwrap();

        unsafe {
            graph
                .call_lazily::<CPU>(&[Some(out.id())], &mut outs_unordered)
                .unwrap()
        }
    }

    #[test]
    fn test_lazy_graph_exec_with_vecs() {
        let mut graph = LazyGraph::default();

        {
            let vec = vec![1, 2, 3, 4];
            graph.add_operation::<u8, CPU, (), _, 1>(vec.no_id(), |_, vec| {
                assert_eq!(vec.as_slice(), &[1, 2, 3, 4]);
                Ok(())
            }).unwrap();
        }
        unsafe { graph.call_lazily::<CPU>(&[None], &mut HashMap::default()) }.unwrap();
    }
}

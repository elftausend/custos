use super::ty::Graphable;
use crate::{
    bounds_to_range, Buffer, Device, DeviceError, Id, NoHasher, Parents, PtrConv, Shape,
    UniqueId,
};
use core::{
    alloc::Layout,
    any::Any,
    hash::BuildHasherDefault,
    mem::{align_of, size_of, transmute},
    ops::RangeBounds,
};
use std::collections::HashMap;

#[derive(Default)]
pub struct LazyGraph {
    pub ids_to_check: Vec<Vec<UniqueId>>,
    pub ops: Vec<fn(*mut (), *mut ()) -> crate::Result<()>>,
    pub args: Vec<*mut ()>,
    pub arg_dealloc_info: Vec<(usize, usize)>,
}

impl Drop for LazyGraph {
    fn drop(&mut self) {
        for (arg_ptr, (align, size)) in self.args.iter().zip(&self.arg_dealloc_info) {
            println!("{align}, {size}");
            let layout = Layout::from_size_align(*size, *align).unwrap();
            if layout.size() != 0 {
                unsafe { std::alloc::dealloc(*arg_ptr as *mut u8, layout) }
            }
        }
    }
}

impl LazyGraph {
    // TODO: could use a broader range of Args! (limited to Parents<N>)
    pub fn add_operation<T, D, S, Args: Parents<N>, const N: usize>(
        &mut self,
        args: Args,
        op: fn(&mut Option<&mut Buffer<T, D, S>>, &mut Args) -> crate::Result<()>,
    ) where
        T: Graphable,
        D: PtrConv,
        S: Shape,
    {
        self.arg_dealloc_info
            .push((align_of::<Args>(), size_of::<Args>()));

        let args = Box::leak(Box::new(args));

        // store ids and test if buffers are still in cache
        self.ids_to_check.push(
            args.maybe_ids()
                .into_iter()
                .flatten()
                .map(|id| *id)
                .collect(),
        );

        self.args.push(args as *mut Args as *mut _);
        unsafe { self.ops.push(transmute(op)) }
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
                    .ok_or(DeviceError::InvalidLazyOutBuf)?;
            }
            match out_id {
                Some(out_id) => {
                    let mut val = outs_unordered.get_mut(out_id).map(|out| &mut **out);

                    let out = &mut val as *mut _ as *mut ();
                    op(out, *args)?;
                }
                None => {
                    let mut val = None::<*mut ()>;
                    let out = &mut val as *mut _ as *mut ();
                    op(out, *args)?;
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
        for ((((args, op), ids_to_check), out_id), (align, size)) in self
            .args
            .drain(range.clone())
            .zip(self.ops.drain(range.clone()))
            .zip(self.ids_to_check.drain(range.clone()))
            .zip(out_buf_order.drain(range.clone()))
            .zip(self.arg_dealloc_info.drain(range))
        {
            for id_to_check in ids_to_check.iter() {
                outs_unordered
                    .get(id_to_check)
                    .ok_or(DeviceError::InvalidLazyOutBuf)?;
            }
            
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

            let layout = Layout::from_size_align(size, align).unwrap();
            if layout.size() != 0 {
                unsafe { std::alloc::dealloc(args as *mut u8, layout) }
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
            });

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
        });

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
        });

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
        });

        unsafe {
            graph
                .call_lazily::<CPU>(&[Some(out.id())], &mut outs_unordered)
                .unwrap()
        }
    }
}

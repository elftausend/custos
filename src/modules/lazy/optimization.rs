use crate::{DeviceError, Lazy, Operation, op_hint::OpHint};

impl<T, Mods> Lazy<'_, Mods, T> {
    pub(crate) unsafe fn alloc_later_optimized<D: 'static>(
        &self,
        device: &D,
        graph_trans: &crate::GraphTranslator,
    ) -> crate::Result<()> {
        let cache_traces = graph_trans.opt_graph.cache_traces();
        for (id, alloc_fn) in self.alloc_later.borrow_mut().drain(..) {
            for cache_trace in &cache_traces {
                let buf_id = graph_trans
                    .idx_to_buf_id
                    .get(&cache_trace.cache_idx)
                    .ok_or(DeviceError::GraphOptimization)?;

                if *buf_id != id.id {
                    continue;
                }

                alloc_fn(
                    &mut self.buffers.borrow_mut(),
                    &mut self.allocated_ids.borrow_mut(),
                    id,
                    device,
                );
                let buf = unsafe { self.buffers.borrow().get(&id.id).unwrap().shallow_copy() };

                // TODO: add type check - lower assert_eq to debug in lazy replace buf

                for use_id_as_well in &cache_trace.use_cache_idxs {
                    let use_id_as_well_id = graph_trans
                        .idx_to_buf_id
                        .get(use_id_as_well)
                        .ok_or(DeviceError::GraphOptimization)?;

                    self.buffers
                        .borrow_mut()
                        .insert(*use_id_as_well_id, unsafe { buf.shallow_copy() });
                }
            }
        }
        Ok(())
    }

    pub(super) fn fuse_unary_ops<D>(
        &self,
        device: &D,
        graph_trans: &crate::GraphTranslator,
    ) -> crate::Result<()>
    where
        D: crate::UnaryFusing + 'static,
        T: crate::Numeric + crate::CDatatype,
    {
        let cache_traces = graph_trans.opt_graph.cache_traces();
        let cache_traces = graph_trans.to_cursor_cache_traces(cache_traces);
        let mut graph = self.graph.borrow_mut();
        let ops = &graph.operations;

        let unary_ops = cache_traces
            .into_iter()
            .map(|mut cache_trace| {
                let mut ids = vec![cache_trace.cache_idx];

                ids.append(&mut cache_trace.use_cache_idxs);
                (
                    ids.iter()
                        .map_while(|id| match &ops[*id].op_hint {
                            OpHint::Unary(op) => Some(op.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                    ids,
                )
            })
            .collect::<Vec<_>>();

        let mut buffers = self.buffers.borrow_mut();

        for unary_ops in unary_ops {
            // skip first and last as first and last nodes are used for fusing
            for to_no_op_idx in unary_ops.1[1..unary_ops.1.len() - 1].iter() {
                graph.operations[*to_no_op_idx] = Operation::no_op();
            }
            let last_idx = *unary_ops.1.last().unwrap();

            // safety: only unary ops are fused. Adding unary ops to the graph require type T.
            let (update_idx, op) =
                unsafe { device.fuse_unary_ops(&graph, unary_ops, &mut buffers) };
            graph.operations[update_idx] = op;
            // only arg id information is required for last op, not the op itself
            graph.operations[last_idx] = Operation::no_op();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    #[test]
    fn test_op_hint_unary_chain_fuse_graph() {
        use crate::{
            AddOperation, ApplyFunction, Base, Buffer, CPU, Combiner, Device, Graph, Lazy,
            Optimize, Retriever, Run,
        };

        let dev = CPU::<Graph<Lazy<Base>>>::new();

        let buf = dev.buffer([1., 2., 3., 4., 5.]);

        let mut out: Buffer<f32, _> = dev.retrieve(buf.len(), &buf).unwrap();
        let mut out1: Buffer<f32, _> = dev.retrieve(buf.len(), &out).unwrap();
        let mut out2: Buffer<f32, _> = dev.retrieve(buf.len(), &out1).unwrap();
        dev.add_op(
            (&buf, &mut out, &mut out1, &mut out2),
            |(buf, out, out1, out2)| {
                // let out = out.as_mut_slice();
                // let out1 = out1.as_mut_slice();
                // for (x, y) in out.iter_mut().zip(out1) {
                //     *y = *x+1.;
                // }
                Ok(())
            },
        )
        .unwrap();

        // unsafe { dev.optimize_mem_graph(&dev, None).unwrap(); }
        dev.run().unwrap();

        // for (buf, out) in buf.iter().zip(_out.replace().iter()) {
        //     assert!((*out - buf.sin().cos().ln()).abs() < 0.001);
        // }
    }
}

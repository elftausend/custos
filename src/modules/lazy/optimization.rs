use crate::{op_hint::OpHint, DeviceError, Lazy, Operation};

impl<T, Mods> Lazy<Mods, T> {
    pub(crate) fn alloc_later_optimized<D: 'static>(
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
                let buf = self.buffers.borrow().get(&id.id).unwrap().shallow_copy();

                // TODO: add type check - lower assert_eq to debug in lazy replace buf

                for use_id_as_well in &cache_trace.use_cache_idxs {
                    let use_id_as_well_id = graph_trans
                        .idx_to_buf_id
                        .get(use_id_as_well)
                        .ok_or(DeviceError::GraphOptimization)?;

                    self.buffers
                        .borrow_mut()
                        .insert(*use_id_as_well_id, buf.shallow_copy());
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

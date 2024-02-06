use core::hash::BuildHasherDefault;
use std::collections::HashMap;

use crate::{CacheTrace, NoHasher, Parents, UniqueId};

use super::opt_graph::OptGraph;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphTranslator {
    pub buf_id_to_idx: HashMap<UniqueId, usize, BuildHasherDefault<NoHasher>>,
    pub idx_to_buf_id: HashMap<usize, UniqueId, BuildHasherDefault<NoHasher>>,
    // As only non-leafs can be located in a CacheTrace, this contains buffers created via retrieving.
    pub idx_to_cursor: HashMap<usize, UniqueId>,
    pub next_idx: usize,
    pub opt_graph: OptGraph,
}

impl GraphTranslator {
    pub fn add_node_type(&mut self, mut node_type_fn: impl FnMut(&mut GraphTranslator)) {
        node_type_fn(self);
        self.next_idx = self.opt_graph.nodes.len();
    }

    pub fn add_leaf(&mut self, len: usize) {
        self.opt_graph.add_node(len, vec![]);
        self.next_idx = self.opt_graph.nodes.len();
    }

    pub fn add_node<const NUM_PARENTS: usize>(
        &mut self,
        len: usize,
        deps: &impl Parents<NUM_PARENTS>,
    ) {
        self.add_node_type(|graph_trans| {
            let deps = deps
                .ids()
                .into_iter()
                .map(|id| *graph_trans.buf_id_to_idx.get(&id).unwrap())
                .collect::<Vec<_>>();

            graph_trans.opt_graph.add_node(len, deps);
        });
    }

    pub fn to_cursor_cache_traces(
        &self,
        cache_traces: Vec<CacheTrace>,
    ) -> Vec<CacheTrace> {
        cache_traces
            .into_iter()
            .map(|cache_trace| CacheTrace {
                cache_idx: *self
                    .idx_to_cursor
                    .get(&cache_trace.cache_idx)
                    .unwrap() as usize,
                use_cache_idxs: cache_trace
                    .use_cache_idxs
                    .into_iter()
                    .map(|cache_idx| *self.idx_to_cursor.get(&cache_idx).unwrap() as usize)
                    .collect(),
            })
            .collect::<Vec<_>>()
    }
}

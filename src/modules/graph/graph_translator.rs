use core::{hash::BuildHasherDefault, panic::Location};
use std::collections::{HashMap, HashSet};

use crate::{CacheTrace, HashLocation, NoHasher, Parents, TranslatedCacheTrace, UniqueId};

use super::opt_graph::OptGraph;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphTranslator {
    pub buf_id_to_idx: HashMap<UniqueId, usize, BuildHasherDefault<NoHasher>>,
    // As only non-leafs can be located in a CacheTrace, this contains buffers created via retrieving.
    pub idx_to_buf_location: HashMap<usize, HashLocation<'static>>,
    pub added_to_graph: HashSet<HashLocation<'static>>,
    pub next_idx: usize,
    pub opt_graph: OptGraph,
}

impl GraphTranslator {
    #[track_caller]
    pub fn add_node_type(&mut self, mut node_type_fn: impl FnMut(&mut GraphTranslator)) {
        if self.added_to_graph.contains(&Location::caller().into()) {
            return;
        }

        self.added_to_graph.insert(Location::caller().into());

        node_type_fn(self);
        self.next_idx = self.opt_graph.nodes.len();
    }

    #[track_caller]
    pub fn add_leaf(&mut self, len: usize) {
        self.opt_graph.add_node(len, vec![]);
        self.next_idx = self.opt_graph.nodes.len();
    }

    #[track_caller]
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

    pub fn translate_cache_traces(
        &self,
        cache_traces: Vec<CacheTrace>,
    ) -> Vec<TranslatedCacheTrace> {
        cache_traces
            .into_iter()
            .map(|cache_trace| TranslatedCacheTrace {
                cache_idx: *self
                    .idx_to_buf_location
                    .get(&cache_trace.cache_idx)
                    .unwrap(),
                use_cache_idxs: cache_trace
                    .use_cache_idxs
                    .into_iter()
                    .map(|cache_idx| *self.idx_to_buf_location.get(&cache_idx).unwrap())
                    .collect(),
            })
            .collect::<Vec<_>>()
    }
}

use core::{hash::BuildHasherDefault, panic::Location};
use std::collections::{HashMap, HashSet};

use crate::{HashLocation, NoHasher, Parents, UniqueId};

use super::opt_graph::OptGraph;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphTranslator {
    pub buf_id_to_idx: HashMap<UniqueId, usize, BuildHasherDefault<NoHasher>>,
    pub added_to_graph: HashSet<HashLocation<'static>>,
    pub next_idx: usize,
    pub opt_graph: OptGraph,
}

impl GraphTranslator {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

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
}

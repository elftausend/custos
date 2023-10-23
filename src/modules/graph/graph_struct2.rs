use core::{hash::BuildHasherDefault, panic::Location};
use std::collections::{HashMap, HashSet};

use crate::{HashLocation, Id, LocationHasher, NoHasher, Parents, UniqueId};

use super::node::Node;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OptGraph {
    pub nodes: Vec<Node>,
}

impl OptGraph {
    /// Adds a leaf node to the graph.
    pub fn add_leaf(&mut self, len: usize) {
        let idx = self.nodes.len();
        let node = Node {
            idx,
            deps: vec![],
            len,
        };
        self.nodes.push(node);
    }

    /// Adds a node to the graph using lhs_idx and rhs_idx as dependencies.
    pub fn add_node(&mut self, len: usize, deps: Vec<usize>) {
        let idx = self.nodes.len();
        let node = Node { idx, deps, len };
        self.nodes.push(node);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphTranslator {
    pub buf_id_to_idx: HashMap<UniqueId, usize>,
    added_to_graph: HashSet<HashLocation<'static>>,
    pub next_idx: usize,
    opt_graph: OptGraph,
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
        self.add_node_type(|graph_trans| graph_trans.opt_graph.add_leaf(len));
    }

    #[track_caller]
    pub fn add_node(&mut self, len: usize, deps: impl Parents<2>) {
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

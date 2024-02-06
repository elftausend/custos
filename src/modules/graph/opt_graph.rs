mod optimize;

use core::hash::BuildHasherDefault;
use std::collections::HashSet;

pub use optimize::*;

use crate::{NoHasher, UniqueId};

use super::node::Node;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OptGraph {
    pub nodes: Vec<Node>,
    pub contains_ids: HashSet<UniqueId, BuildHasherDefault<NoHasher>>
}

impl OptGraph {
    /// Adds a leaf node to the graph.
    pub fn add_leaf(&mut self, len: usize) -> usize {
        let idx = self.nodes.len();
        let node = Node {
            idx,
            deps: vec![],
            len,
        };
        self.nodes.push(node);
        idx
    }

    /// Adds a node to the graph using lhs_idx and rhs_idx as dependencies.
    pub fn add_node(&mut self, len: usize, deps: Vec<usize>) -> usize {
        let idx = self.nodes.len();
        let node = Node { idx, deps, len };
        self.nodes.push(node);
        idx
    }

    pub fn node(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }
}

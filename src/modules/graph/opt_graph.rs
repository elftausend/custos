mod optimize;

use super::node::Node;
pub use optimize::*;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OptGraph {
    pub nodes: Vec<Node>,
}

impl OptGraph {
    /// Adds a leaf node to the graph.
    pub fn add_leaf(&mut self, len: usize) -> usize {
        let idx = self.nodes.len();
        let node = Node {
            idx,
            deps: vec![],
            len,
            skip: false,
        };
        self.nodes.push(node);
        idx
    }

    /// Adds a node to the graph using lhs_idx and rhs_idx as dependencies.
    pub fn add_node(&mut self, len: usize, deps: Vec<usize>) -> usize {
        let idx = self.nodes.len();
        let node = Node { idx, deps, len, skip: false };
        self.nodes.push(node);
        idx
    }

    pub fn node(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }
}

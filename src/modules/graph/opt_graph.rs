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

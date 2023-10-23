use super::{graph_struct2::GraphTranslator, node::Node};
use crate::Parents;

/// Trait for adding a node to a graph.
pub trait AddToGraph<const N: usize>: Parents<N> {
    /// Adds a node to the graph.
    #[track_caller]
    #[inline]
    fn add(&self, graph: &mut GraphTranslator, len: usize) -> Node {
        let ids = self.ids();
        todo!()
        // graph.add_node(len, lhs_idx, rhs_idx)
    }
}

impl AddToGraph<0> for () {
    #[inline]
    fn add(&self, graph: &mut GraphTranslator, len: usize) -> Node {
        todo!()
        // graph.add_leaf(len)
    }
}

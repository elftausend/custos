use crate::{shape::Shape, Buffer, Device, Graph};

use super::node::Node;

pub trait AddGraph {
    fn add(&self, graph: &mut Graph, len: usize) -> Node;
}

impl AddGraph for () {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_leaf(len)
    }
}

// Unary operation
impl AddGraph for usize {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, *self, *self)
    }
}

impl AddGraph for (usize, usize) {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0, self.1)
    }
}

impl AddGraph for [usize; 2] {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0], self[1])
    }
}

impl AddGraph for [usize; 1] {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0], self[0])
    }
}

impl<'a, T, D: Device, S: Shape> AddGraph for Buffer<'a, T, D, S> {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.ident.idx, self.ident.idx)
    }
}

impl<'a, T, D: Device, S: Shape> AddGraph for &Buffer<'a, T, D, S> {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.ident.idx, self.ident.idx)
    }
}

impl<'a, T, D: Device, LS: Shape, RS: Shape> AddGraph
    for (&Buffer<'a, T, D, LS>, &Buffer<'a, T, D, RS>)
{
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0.ident.idx, self.1.ident.idx)
    }
}

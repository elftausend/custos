use crate::{Buffer, Device, Graph};

use super::node::Node;

pub trait AddGraph {
    fn add(&self, graph: &mut Graph, len: usize) -> Node;
}

impl AddGraph for () {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_leaf(len)
    }
}

// Unary operation
impl AddGraph for usize {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, *self as isize, *self as isize)
    }
}

// Unary operation
impl AddGraph for isize {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, *self as isize, *self)
    }
}

impl AddGraph for (usize, usize) {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0 as isize, self.1 as isize)
    }
}

impl AddGraph for (isize, isize) {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0, self.1)
    }
}

impl AddGraph for [usize; 2] {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0] as isize, self[1] as isize)
    }
}

impl AddGraph for [isize; 2] {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0], self[1])
    }
}

impl AddGraph for [usize; 1] {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0] as isize, self[0] as isize)
    }
}

pub struct CachedLeaf;

impl AddGraph for CachedLeaf {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, -1, -1)
    }
}

impl<'a, T, D: Device, const N: usize> AddGraph for Buffer<'a, T, D, N> {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.node.idx, self.node.idx)
    }
}

impl<'a, T, D: Device, const N: usize> AddGraph for &Buffer<'a, T, D, N> {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.node.idx, self.node.idx)
    }
}

impl<'a, T, D: Device, const N: usize> AddGraph for (&Buffer<'a, T, D, N>, &Buffer<'a, T, D, N>) {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0.node.idx, self.1.node.idx)
    }
}

impl<'a, T, D: Device, const N: usize> AddGraph for [&Buffer<'a, T, D, N>; 2] {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0].node.idx, self[1].node.idx)
    }
}

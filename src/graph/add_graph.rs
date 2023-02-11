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
        graph.add_node(len, *self as isize, *self as isize)
    }
}

// Unary operation
impl AddGraph for isize {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, *self, *self)
    }
}

impl AddGraph for (usize, usize) {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0 as isize, self.1 as isize)
    }
}

impl AddGraph for (isize, isize) {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0, self.1)
    }
}

impl AddGraph for [usize; 2] {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0] as isize, self[1] as isize)
    }
}

impl AddGraph for [isize; 2] {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0], self[1])
    }
}

impl AddGraph for [usize; 1] {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0] as isize, self[0] as isize)
    }
}

pub struct CachedLeaf;

impl AddGraph for CachedLeaf {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, -1, -1)
    }
}

impl<'a, T, D: Device, S: Shape> AddGraph for Buffer<'a, T, D, S> {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.ident.idx as isize, self.ident.idx as isize)
    }
}

impl<'a, T, D: Device, S: Shape> AddGraph for &Buffer<'a, T, D, S> {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.ident.idx as isize, self.ident.idx as isize)
    }
}

impl<'a, T, D: Device, S: Shape> AddGraph for (&Buffer<'a, T, D, S>, &Buffer<'a, T, D, S>) {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0.ident.idx as isize, self.1.ident.idx as isize)
    }
}

impl<'a, T, D: Device, S: Shape> AddGraph for [&Buffer<'a, T, D, S>; 2] {
    #[inline]
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self[0].ident.idx as isize, self[1].ident.idx as isize)
    }
}

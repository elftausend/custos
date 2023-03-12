use core::{hash::BuildHasherDefault, marker::PhantomData};
use std::collections::{HashMap, HashSet};

use crate::{get_count, AddGraph, CacheTrace, Ident, IdentHasher, Node};

/// A graph of [`Node`]s.
/// It is typically built up during the forward process. (calling `device.retrieve(.., (lhs, rhs))`)
#[derive(Default, Debug)]
pub struct Graph<IdxFrom: NodeIdx> {
    pub nodes: Vec<Node>,
    /// Translates the index to a [`Node`] in the graph, to an index in the cache / global count.
    pub idx_trans: HashMap<usize, usize, BuildHasherDefault<IdentHasher>>,
    _pd: PhantomData<IdxFrom>,
}

/// Returns the next index for a [`Node`].
pub trait NodeIdx {
    /// Returns the next index for a [`Node`].
    #[inline]
    fn idx(nodes: &[Node]) -> usize {
        nodes.len()
    }
}

/// Uses the global count as the next index for a [`Node`].
#[derive(Debug, Default)]
pub struct GlobalCount;

impl NodeIdx for GlobalCount {
    #[inline]
    fn idx(_nodes: &[Node]) -> usize {
        get_count()
    }
}

/// Uses the amount of nodes in the graph as the next index for a [`Node`].
#[derive(Debug, Default)]
pub struct NodeCount;

impl NodeIdx for NodeCount {}

impl<IdxFrom: NodeIdx> Graph<IdxFrom> {
    /// Creates a new graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            idx_trans: HashMap::default(),
            _pd: PhantomData,
        }
    }

    /// Adds a node to the graph.
    #[inline]
    pub fn add(&mut self, len: usize, add_node: impl AddGraph) -> Node {
        add_node.add(self, len)
    }

    pub fn add_leaf(&mut self, len: usize) -> Node {
        let idx = self.nodes.len();
        let ident_idx = IdxFrom::idx(&self.nodes);
        let node = Node {
            //ident_idx: idx,
            idx,
            deps: [idx, idx],
            len,
        };
        self.nodes.push(node);
        self.idx_trans.insert(idx, ident_idx);
        node
    }

    pub fn add_node(&mut self, len: usize, lhs_idx: usize, rhs_idx: usize) -> Node {
        let idx = self.nodes.len();
        let ident_idx = IdxFrom::idx(&self.nodes);
        let node = Node {
            // ident_idx: idx,
            idx,
            deps: [lhs_idx, rhs_idx],
            len,
        };
        self.nodes.push(node);
        self.idx_trans.insert(idx, ident_idx);
        node
    }

    /// Calculates multiple unique [`CacheTrace`]s.
    /// Unique meaning that no two [`CacheTrace`]s share some same [`Node`].
    pub fn cache_traces(&self) -> Vec<CacheTrace> {
        let mut traces = vec![];
        let mut visited_nodes = HashSet::new();

        for node in self.nodes.iter().filter(|node| !node.is_leaf()) {
            if visited_nodes.contains(node) {
                continue;
            }

            let trace = self.trace_cache_path(node);

            if trace.is_empty() {
                continue;
            }

            traces.push(CacheTrace {
                cache_idx: node.idx,
                use_cache_idx: trace
                    .into_iter()
                    //.filter(|node| !visited_nodes.contains(*node))
                    .map(|node| {
                        visited_nodes.insert(node);
                        Ident {
                            idx: *self.idx_trans.get(&node.idx).unwrap(),
                            len: node.len,
                        }
                    })
                    .collect(),
            });
        }

        traces
    }

    /// Calculates the cache trace for a starting node.
    /// A cache trace is a list of nodes that shows which [`Buffer`](crate::Buffer)s could use the same cache.
    pub fn trace_cache_path(&self, trace_at: &Node) -> Vec<Node> {
        if !self.is_path_optimizable(trace_at) {
            return vec![];
        }

        let mut trace = vec![];
        let mut idx = trace_at.idx;

        for check in self.nodes.iter().skip(trace_at.idx + 1) {
            if !check.deps.contains(&idx) {
                continue;
            }

            if trace_at.len != check.len {
                continue;
            }

            idx = check.idx;
            trace.push(*check);

            // the first unoptimizable node in a cache trace may be added to the cache trace
            // look test "test_cache_trace_break_not_anymore"
            if !self.is_path_optimizable(check) {
                break;
            }
        }

        trace
    }

    /// A path is optimizable, if it contains only one occurence of the starting node.
    /// # Example
    /// ```
    /// use custos::{Graph, NodeCount};
    /// 
    /// let mut graph = Graph::<NodeCount>::new();
    /// let a = graph.add_leaf(10);
    /// let b = graph.add_leaf(10);
    ///     
    /// let c = graph.add_node(10, a.idx, b.idx);
    ///     
    /// let d = graph.add_node(10, c.idx, c.idx);
    ///     
    /// let _u = graph.add_node(10, d.idx, a.idx);
    ///     
    /// let _e = graph.add_node(10, d.idx, b.idx);
    ///     
    /// assert!(graph.is_path_optimizable(&c));
    /// assert!(!graph.is_path_optimizable(&d));
    /// ```
    pub fn is_path_optimizable(&self, check_at: &Node) -> bool {
        if check_at.is_leaf() {
            return false;
        };

        let mut occurences = 0;

        for check in &self.nodes[check_at.idx + 1..] {
            if check_at.len != check.len || !check.deps.contains(&check_at.idx) {
                continue;
            }

            if occurences >= 1 {
                return false;
            }
            occurences += 1;
        }

        true
    }
}

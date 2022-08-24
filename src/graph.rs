use std::cell::RefMut;

use crate::{
    cache::{CacheReturn, CacheType},
    Node,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheTrace {
    pub cache_idx: usize,
    pub use_cache_idx: Vec<Node>,
}

pub trait GraphReturn {
    fn graph(&self) -> RefMut<Graph>;
}

pub trait GraphOpt {
    fn optimize<P>(&self)
    where
        P: CacheType,
        Self: GraphReturn + CacheReturn<P>,
    {
        let mut cache = self.cache();

        let cache_traces = self.graph().cache_traces();

        if let Some(cache_traces) = &cache_traces {
            for trace in cache_traces {

                //let node = cache.nodes.get()

                /*cache.nodes.remove(
                    &trace.cache_idx
                );*/
            }
        }

        cache.cache_traces = cache_traces
    }
}

#[derive(Default, Debug)]
pub struct Graph {
    nodes: Vec<GNode>,
    leafs: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            leafs: 0,
        }
    }

    pub fn add<A: AddGraph>(&mut self, len: usize, add_node: A) -> GNode {
        add_node.add(self, len)
    }

    pub fn add_leaf(&mut self, len: usize) -> GNode {
        let idx = self.nodes.len();
        self.leafs += 1;
        self.add_node(len, idx, idx)
    }

    pub fn add_node(&mut self, len: usize, lhs_idx: usize, rhs_idx: usize) -> GNode {
        let idx = self.nodes.len();
        let node = GNode {
            idx,
            deps: [lhs_idx, rhs_idx],
            len,
        };
        self.nodes.push(node);
        node
    }

    pub fn cache_traces(&self) -> Option<Vec<CacheTrace>> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut start = GNode::default();
        for node in &self.nodes {
            if !node.is_leaf() {
                start = *node;
                break;
            }
        }

        let offset = start.idx;
        let mut traces = vec![];

        while let Some(trace) = self.trace_cache_path(&start) {
            let last_trace_node = *trace.last().unwrap();

            traces.push(CacheTrace {
                cache_idx: start.idx - offset,
                use_cache_idx: trace
                    .into_iter()
                    .map(|node| Node {
                        idx: node.idx - offset,
                        len: node.len,
                    })
                    .collect(),
            });
            if let Some(last) = self.nodes.get(last_trace_node.idx + 1) {
                start = *last;
            } else {
                break;
            }
        }
        Some(traces)
    }

    pub fn trace_cache_path(&self, trace_at: &GNode) -> Option<Vec<GNode>> {
        if !self.is_path_optimizable(trace_at) {
            return None;
        }

        let mut trace = vec![*trace_at];

        let mut idx = trace_at.idx;
        for check in &self.nodes[trace_at.idx + 1..] {
            if trace_at.len != check.len || !self.is_path_optimizable(check) {
                continue;
            }
            if check.deps.contains(&idx) {
                idx = check.idx;
                trace.push(*check);
            }
        }
        Some(trace)
    }

    pub fn is_path_optimizable(&self, check_at: &GNode) -> bool {
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

    pub fn is_optimizable(&mut self) -> bool {
        for node in &self.nodes {
            if node.is_leaf() {
                continue;
            }

            let path_optimizable = self.is_path_optimizable(node);

            println!("node: {node:?}, is_opt: {path_optimizable}");
            //if path_optimizable { return true }
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GNode {
    pub idx: usize,
    pub deps: [usize; 2],
    pub len: usize,
}

impl GNode {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.idx == (self.deps[0] & self.deps[1])
    }
}

pub trait AddGraph {
    fn add(&self, graph: &mut Graph, len: usize) -> GNode;
}

// Leaf cache
impl AddGraph for () {
    fn add(&self, graph: &mut Graph, len: usize) -> GNode {
        graph.add_leaf(len)
    }
}

// Unary operation
impl AddGraph for usize {
    fn add(&self, graph: &mut Graph, len: usize) -> GNode {
        graph.add_node(len, *self, *self)
    }
}

impl AddGraph for (usize, usize) {
    fn add(&self, graph: &mut Graph, len: usize) -> GNode {
        graph.add_node(len, self.0, self.1)
    }
}

impl AddGraph for [usize; 2] {
    fn add(&self, graph: &mut Graph, len: usize) -> GNode {
        graph.add_node(len, self[0], self[1])
    }
}
impl AddGraph for [usize; 1] {
    fn add(&self, graph: &mut Graph, len: usize) -> GNode {
        graph.add_node(len, self[0], self[0])
    }
}

#[cfg(test)]
mod tests {
    use crate::{CacheTrace, GNode, Graph, Node};

    #[test]
    fn test_leaf_node() {
        let node = GNode {
            idx: 1,
            deps: [1, 1],
            len: 10,
        };
        assert!(node.is_leaf());

        let node = GNode {
            idx: 2,
            deps: [1, 1],
            len: 10,
        };
        assert!(!node.is_leaf());

        let node = GNode {
            idx: 2,
            deps: [1, 2],
            len: 10,
        };
        assert!(!node.is_leaf());
    }

    #[test]
    fn test_cache_trace() {
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1]
        let c = graph.add_node(10, a.idx, b.idx);

        // idx: 3, deps: [2, 2]
        let d = graph.add_node(10, c.idx, c.idx);

        // idx: 4, deps: [3, 1]
        let _e = graph.add_node(10, d.idx, b.idx);

        // idx: 5, deps: [2, 1]
        //let f = graph.add_node(10, c.idx, b.idx);

        let trace = graph.trace_cache_path(&c);
        assert_eq!(
            Some(vec![
                GNode {
                    idx: 2,
                    deps: [0, 1],
                    len: 10
                },
                GNode {
                    idx: 3,
                    deps: [2, 2],
                    len: 10
                },
                GNode {
                    idx: 4,
                    deps: [3, 1],
                    len: 10
                }
            ]),
            trace
        );

        let traces = graph.cache_traces();
        println!("traces: {traces:?}");
    }

    #[test]
    fn test_no_cache_trace() {
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1]
        let c = graph.add_node(10, a.idx, b.idx);

        // idx: 3, deps: [2, 2]
        let d = graph.add_node(10, c.idx, c.idx);

        // idx: 4, deps: [3, 1]
        let _e = graph.add_node(10, d.idx, b.idx);

        // idx: 5, deps: [2, 1]
        let _f = graph.add_node(10, c.idx, b.idx);

        let trace = graph.trace_cache_path(&c);
        assert_eq!(None, trace);
    }

    #[test]
    fn test_cache_trace_2() {
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);
        let u = graph.add_leaf(10);

        let c = graph.add_node(10, a.idx, b.idx);

        let _z = graph.add_node(10, a.idx, u.idx);

        let d = graph.add_node(10, c.idx, c.idx);
        let _e = graph.add_node(10, d.idx, b.idx);

        let trace = graph.trace_cache_path(&c);
        assert_eq!(
            Some(vec![
                GNode {
                    idx: 3,
                    deps: [0, 1],
                    len: 10
                },
                GNode {
                    idx: 5,
                    deps: [3, 3],
                    len: 10
                },
                GNode {
                    idx: 6,
                    deps: [5, 1],
                    len: 10
                }
            ]),
            trace
        );
    }

    #[test]
    fn test_cache_trace_break() {
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1]
        let c = graph.add_node(10, a.idx, b.idx);

        // idx: 3, deps: [2, 2]
        let d = graph.add_node(10, c.idx, c.idx);

        // idx: 4, deps: [3, 0]
        let _u = graph.add_node(10, d.idx, a.idx);

        // idx: 5, deps: [3, 1]
        let _e = graph.add_node(10, d.idx, b.idx);

        let _trace = graph.trace_cache_path(&c);

        assert!(graph.is_path_optimizable(&c));
        assert!(!graph.is_path_optimizable(&d));
        //println!("trace: {trace:?}");
    }

    #[test]
    fn test_trace_all() {
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1] (0)
        let c = graph.add_node(10, a.idx, b.idx);

        // idx: 3, deps: [2, 2] (1)
        let d = graph.add_node(10, c.idx, c.idx);

        // idx: 4, deps: [3, 1] (2)
        let _e = graph.add_node(10, d.idx, b.idx);

        let traces = graph.cache_traces();

        assert_eq!(
            CacheTrace {
                cache_idx: 0,
                use_cache_idx: vec![
                    Node { idx: 0, len: 10 },
                    Node { idx: 1, len: 10 },
                    Node { idx: 2, len: 10 },
                ],
            },
            traces.unwrap()[0]
        );
    }
}

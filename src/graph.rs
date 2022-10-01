use crate::{Buffer, Ident, COUNT};
use std::cell::RefMut;

#[cfg(feature = "opt-cache")]
use crate::{cache::CacheReturn, DeviceError};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheTrace {
    pub cache_idx: usize,
    pub use_cache_idx: Vec<Ident>,
}

pub trait GraphReturn {
    fn graph(&self) -> RefMut<Graph>;
}

#[cfg(feature = "opt-cache")]
pub trait GraphOpt {
    fn optimize(&self) -> crate::Result<()>
    where
        Self: GraphReturn + CacheReturn,
    {
        let mut cache = self.cache();

        for trace in self.graph().cache_traces() {
            // starting at 1, because the first element is the origin
            for node in &trace.use_cache_idx[1..] {
                // insert the common / optimized pointer in all the other nodes
                // this deallocates the old pointers
                let ptr = cache
                    .nodes
                    .get(&trace.use_cache_idx[0])
                    .ok_or(DeviceError::GraphOptimization)?
                    .clone();
                cache.nodes.insert(*node, ptr);
            }
        }
        Ok(())
    }
}

#[derive(Default, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add(&mut self, len: usize, add_node: impl AddGraph) -> Node {
        add_node.add(self, len)
    }

    pub fn add_leaf(&mut self, len: usize) -> Node {
        Node {
            idx: -1,
            ident_idx: -1,
            deps: [-1, -1],
            len,
        }
    }

    pub fn add_node(&mut self, len: usize, lhs_idx: isize, rhs_idx: isize) -> Node {
        let idx = self.nodes.len() as isize;
        let node = COUNT.with(|count| {
            Node {
                // subtracting 1, because the count is increased beforehand.
                ident_idx: *count.borrow() as isize,
                idx,
                deps: [lhs_idx, rhs_idx],
                len,
            }
        });
        self.nodes.push(node);
        node
    }

    pub fn cache_traces(&self) -> Vec<CacheTrace> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut start = self.nodes[0];
        let mut traces = vec![];

        while let Some(trace) = self.trace_cache_path(&start) {
            let last_trace_node = *trace.last().unwrap();

            traces.push(CacheTrace {
                cache_idx: start.idx as usize,
                use_cache_idx: trace
                    .into_iter()
                    .map(|node| Ident {
                        idx: node.ident_idx as usize,
                        len: node.len as usize,
                    })
                    .collect(),
            });

            // use better searching algorithm to find the next start node
            match self.nodes.get(last_trace_node.idx as usize + 1) {
                Some(next) => start = *next,
                None => return traces,
            }
        }
        traces
    }

    pub fn trace_cache_path(&self, trace_at: &Node) -> Option<Vec<Node>> {
        if !self.is_path_optimizable(trace_at) {
            return None;
        }

        let mut trace = vec![*trace_at];

        let mut idx = trace_at.idx;
        for check in &self.nodes[trace_at.idx as usize + 1..] {
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

    pub fn is_path_optimizable(&self, check_at: &Node) -> bool {
        if check_at.is_leaf() {
            return false;
        };

        let mut occurences = 0;

        for check in &self.nodes[check_at.idx as usize + 1..] {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node {
    pub ident_idx: isize,
    pub idx: isize,
    pub deps: [isize; 2],
    pub len: usize,
}

impl Default for Node {
    #[inline]
    fn default() -> Self {
        Self {
            ident_idx: -1,
            idx: -1,
            deps: [-1, -1],
            len: 0,
        }
    }
}

impl Node {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.idx == -1
    }
}

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

impl<'a, T, D> AddGraph for Buffer<'a, T, D> {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.node.idx, self.node.idx)
    }
}

impl<'a, T, D> AddGraph for &Buffer<'a, T, D> {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.node.idx, self.node.idx)
    }
}

impl<'a, T, D> AddGraph for (&Buffer<'a, T, D>, &Buffer<'a, T, D>) {
    fn add(&self, graph: &mut Graph, len: usize) -> Node {
        graph.add_node(len, self.0.node.idx, self.1.node.idx)
    }
}

#[cfg(test)]
mod tests {
    use crate::{bump_count, set_count, CacheTrace, Graph, Ident, Node};

    // test if node is a leaf node
    #[test]
    fn test_is_leaf() {
        let mut graph = Graph::new();
        let node = graph.add_leaf(0);
        assert!(node.is_leaf());

        let node = graph.add_node(10, -1, -1);
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
                Node {
                    ident_idx: 0,
                    idx: 0,
                    deps: [-1, -1],
                    len: 10
                },
                Node {
                    ident_idx: 0,
                    idx: 1,
                    deps: [0, 0],
                    len: 10
                },
                Node {
                    ident_idx: 0,
                    idx: 2,
                    deps: [1, -1],
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
                Node {
                    ident_idx: 0,
                    idx: 0,
                    deps: [-1, -1],
                    len: 10
                },
                Node {
                    ident_idx: 0,
                    idx: 2,
                    deps: [0, 0],
                    len: 10
                },
                Node {
                    ident_idx: 0,
                    idx: 3,
                    deps: [2, -1],
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

        let trace = graph.trace_cache_path(&c);

        // TODO: d could use the memory of c, but this is not the case yet
        assert_eq!(
            Some(vec![
                Node {
                    ident_idx: 0,
                    idx: 0,
                    deps: [-1, -1],
                    len: 10
                },
                /* if d uses the memory of c, this node could be added:
                Node {
                    ident_idx: 0,
                    idx: 1,
                    deps: [0, 0],
                    len: 10
                },*/
            ]),
            trace
        );

        assert!(graph.is_path_optimizable(&c));
        assert!(!graph.is_path_optimizable(&d));
    }

    #[test]
    fn test_trace_all() {
        set_count(0);
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1] (0)
        let c = graph.add_node(10, a.idx, b.idx);
        bump_count();

        // idx: 3, deps: [2, 2] (1)
        let d = graph.add_node(10, c.idx, c.idx);
        bump_count();

        // idx: 4, deps: [3, 1] (2)
        let _e = graph.add_node(10, d.idx, b.idx);
        bump_count();

        let traces = graph.cache_traces();

        assert_eq!(
            CacheTrace {
                cache_idx: 0,
                use_cache_idx: vec![
                    Ident { idx: 0, len: 10 },
                    Ident { idx: 1, len: 10 },
                    Ident { idx: 2, len: 10 },
                ],
            },
            traces[0]
        );
    }

    #[test]
    fn test_leafed_diff_len_trace() {
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let _b = graph.add_node(10, a.idx, a.idx);
        bump_count();

        let _z = graph.add_leaf(10);

        let _z = graph.add_leaf(10);

        // idx: 2, deps: [0, 1] (0)
        let c = graph.add_node(12, a.idx, a.idx);
        bump_count();

        // idx: 3, deps: [2, 2] (1)
        let d = graph.add_node(12, c.idx, c.idx);
        bump_count();

        // idx: 4, deps: [3, 1] (2)
        let _e = graph.add_node(12, d.idx, a.idx);
        bump_count();

        let traces = graph.cache_traces();

        assert_eq!(
            CacheTrace {
                cache_idx: 0,
                use_cache_idx: vec![Ident { idx: 0, len: 10 },],
            },
            traces[0]
        );

        assert_eq!(
            CacheTrace {
                cache_idx: 1,
                use_cache_idx: vec![
                    Ident { idx: 1, len: 12 },
                    Ident { idx: 2, len: 12 },
                    Ident { idx: 3, len: 12 },
                ],
            },
            traces[1]
        );
        //        println!("traces: {traces:?}");
    }
}

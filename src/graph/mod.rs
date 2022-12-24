#[cfg(not(feature = "no-std"))]
use crate::Ident;

use core::cell::RefMut;

#[cfg(feature = "opt-cache")]
use crate::{CacheReturn, DeviceError};

pub use add_graph::*;
pub use node::*;

mod add_graph;
mod node;

#[cfg(not(feature = "no-std"))]
mod graph_struct;

#[cfg(not(feature = "no-std"))]
pub use graph_struct::Graph;

#[cfg(feature = "no-std")]
pub struct Graph {}

#[cfg(feature = "no-std")]
impl Graph {
    #[inline]
    pub fn add_leaf(&mut self, len: usize) -> Node {
        Node {
            idx: -1,
            ident_idx: -1,
            deps: [-1, -1],
            len,
        }
    }
    #[inline]
    pub fn add_node(&mut self, len: usize, lhs_idx: isize, rhs_idx: isize) -> Node {
        self.add_leaf(len)
    }
}


#[cfg(not(feature = "no-std"))]
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
        Self: GraphReturn + CacheReturn + RawConv,
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

#[cfg(not(feature = "no-std"))]
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
        // for: cargo test -- --test-threads=1
        set_count(0);
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

        let _traces = graph.cache_traces();
        //println!("traces: {traces:?}");
    }

    #[test]
    fn test_no_cache_trace() {
        // for: cargo test -- --test-threads=1
        set_count(0);
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
        // for: cargo test -- --test-threads=1
        set_count(0);
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
        // for: cargo test -- --test-threads=1
        set_count(0);
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
        // for: cargo test -- --test-threads=1
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
        // for: cargo test -- --test-threads=1
        set_count(0);
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

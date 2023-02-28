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
        Self: GraphReturn + CacheReturn + crate::RawConv,
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
    use crate::{set_count, CacheTrace, Graph, Ident, Node};

    #[test]
    fn test_is_leaf() {
        let mut graph = Graph::new();
        let node = graph.add_leaf(0);
        assert!(node.is_leaf());

        let node = graph.add_node(10, 1, 2);
        assert!(!node.is_leaf());
    }

    #[test]
    fn test_cache_trace() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
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
            vec![
                Node {
                    idx: 3,
                    deps: [2, 2],
                    len: 10,
                },
                Node {
                    idx: 4,
                    deps: [3, 1],
                    len: 10,
                }
            ],
            trace
        );

        let _traces = graph.cache_traces();
        //println!("traces: {traces:?}");
    }

    #[test]
    fn test_no_cache_trace() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
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
        assert_eq!(Vec::<Node>::new(), trace);
    }

    #[test]
    fn test_cache_trace_2() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::new();
        let a = graph.add_leaf(10); // idx: 0
        let b = graph.add_leaf(10); // idx: 1
        let u = graph.add_leaf(10); // idx: 2

        // idx: 3, deps: [0, 1]
        let c = graph.add_node(10, a.idx, b.idx);

        // idx: 4, deps: [0, 2]
        let _z = graph.add_node(10, a.idx, u.idx);

        // idx: 5, deps: [3, 3]
        let d = graph.add_node(10, c.idx, c.idx);

        // idx: 6, deps: [5, 1]
        let _e = graph.add_node(10, d.idx, b.idx);

        let trace = graph.trace_cache_path(&c);
        assert_eq!(
            vec![
                Node {
                    idx: 5,
                    deps: [3, 3],
                    len: 10,
                },
                Node {
                    idx: 6,
                    deps: [5, 1],
                    len: 10,
                }
            ],
            trace
        );
    }

    #[test]
    fn test_cache_trace_break_not_anymore() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
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

        println!("traces: {:?}", graph.cache_traces());

        let trace = graph.trace_cache_path(&c);
        println!("c_trace: {:?}", trace);
        /*assert_eq!(
            vec![
                Node {
                    idx: 2,
                    deps: [0, 0],
                    len: 10
                },
            ],
            trace
        );*/

        assert!(graph.is_path_optimizable(&c));
        assert!(!graph.is_path_optimizable(&d));
    }

    #[test]
    fn test_trace_all() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
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

        assert_eq!(traces.len(), 1);

        assert_eq!(
            CacheTrace {
                cache_idx: 2,
                use_cache_idx: vec![
                    Ident { idx: 3, len: 10 },
                    Ident { idx: 4, len: 10 },
                ],
            },
            traces[0]
        );
    }

    #[test]
    fn test_leafed_diff_len_trace() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);

        let _b = graph.add_node(10, a.idx, a.idx);

        let _z = graph.add_leaf(10);

        let _z = graph.add_leaf(10);

        // idx: 2, deps: [0, 1] (0)
        let c = graph.add_node(12, a.idx, a.idx);

        // idx: 3, deps: [2, 2] (1)
        let d = graph.add_node(12, c.idx, c.idx);

        // idx: 4, deps: [3, 1] (2)
        let _e = graph.add_node(12, d.idx, a.idx);

        let traces = graph.cache_traces();

        assert_eq!(
            [CacheTrace {
                cache_idx: 4,
                use_cache_idx: vec![Ident { idx: 5, len: 12 }, Ident { idx: 6, len: 12 },],
            }],
            &*traces
        );
    }

    #[test]
    fn test_cache_trace_neural_net() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::new();
        let inputs = graph.add_leaf(100 * 10);
        let targets = graph.add_leaf(100);

        let w1 = graph.add_leaf(10 * 64);
        let b1 = graph.add_leaf(64);
        let w2 = graph.add_leaf(64 * 64);
        let b2 = graph.add_leaf(64);
        let w3 = graph.add_leaf(64 * 64);
        let b3 = graph.add_leaf(64);
        let w4 = graph.add_leaf(64 * 1);
        let b4 = graph.add_leaf(1);

        let a1 = graph.add_node(100 * 64, inputs.idx, w1.idx);
        let a2 = graph.add_node(100 * 64, a1.idx, b1.idx);
        let a2 = graph.add_node(100 * 64, a2.idx, a2.idx);

        let a3 = graph.add_node(100 * 64, a2.idx, w2.idx);
        let a4 = graph.add_node(100 * 64, a3.idx, b2.idx);
        let a4 = graph.add_node(100 * 64, a4.idx, a4.idx);

        let a5 = graph.add_node(100 * 64, a4.idx, w3.idx);
        let a6 = graph.add_node(100 * 64, a5.idx, b3.idx);
        let a6 = graph.add_node(100 * 64, a6.idx, a6.idx);
        let a7 = graph.add_node(100 * 1, a6.idx, w4.idx);
        let a8 = graph.add_node(100 * 1, a7.idx, b4.idx);

        let _loss = graph.add_node(100, a8.idx, targets.idx);

        let traces = graph.cache_traces();
        assert_eq!(
            traces,
            [
                CacheTrace {
                    cache_idx: 10,
                    use_cache_idx: vec![
                        //   Ident { idx: 0, len: 6400 },
                        Ident { idx: 11, len: 6400 },
                        Ident { idx: 12, len: 6400 },
                        Ident { idx: 13, len: 6400 },
                        Ident { idx: 14, len: 6400 },
                        Ident { idx: 15, len: 6400 },
                        Ident { idx: 16, len: 6400 },
                        Ident { idx: 17, len: 6400 },
                        Ident { idx: 18, len: 6400 }
                    ]
                },
                CacheTrace {
                    cache_idx: 19,
                    use_cache_idx: vec![
                        //   Ident { idx: 0, len: 6400 },
                        Ident { idx: 20, len: 100 },
                        Ident { idx: 21, len: 100 },
                    ]
                }
            ]
        )

        // graph.add_node(10*10, gemm.idx, gemm.idx);
        // bump_count();
    }

    #[test]
    fn test_cache_trace_d() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1]
        let c = graph.add_node(10, a.idx, b.idx);

        // idx: 3, deps: [2, 2]
        let d = graph.add_node(10, c.idx, c.idx);

        // idx: 4, deps: [3, 0]
        let _u = graph.add_node(10, a.idx, d.idx);

        // idx: 5, deps: [3, 1]
        //let _e = graph.add_node(10, d.idx, b.idx);

        let trace = graph.trace_cache_path(&c);

        assert_eq!(
            vec![
                Node {
                    idx: 3,
                    deps: [2, 2],
                    len: 10,
                },
                Node {
                    idx: 4,
                    deps: [0, 3],
                    len: 10,
                }
            ],
            trace
        );

        assert!(graph.is_path_optimizable(&c));
        assert!(graph.is_path_optimizable(&d));

        let trace = graph.cache_traces();
        assert_eq!(
            trace,
            [CacheTrace {
                cache_idx: 2,
                use_cache_idx: vec![Ident { idx: 3, len: 10 }, Ident { idx: 4, len: 10 }]
            }]
        );
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "opt-cache")]
    #[test]
    fn test_from_retrieve() {
        use crate::{Buffer, Device, GraphReturn, CPU};

        let device = CPU::new();

        let w1 = Buffer::from((&device, [1; 10 * 64]));
        let b1 = Buffer::from((&device, [1; 64]));
        let w2 = Buffer::from((&device, [1; 64 * 64]));
        let b2 = Buffer::from((&device, [1; 64]));
        let w3 = Buffer::from((&device, [1; 64 * 64]));
        let b3 = Buffer::from((&device, [1; 64]));
        let w4 = Buffer::from((&device, [1; 64 * 1]));
        let b4 = Buffer::from((&device, [1; 1]));

        let inputs = Buffer::from((&device, [1; 10 * 100]));
        let targets = Buffer::from((&device, [2; 100]));

        let a1 = device.retrieve::<i32, ()>(100 * 64, (&inputs, &w1));
        let a2 = device.retrieve::<i32, ()>(100 * 64, (&a1, &b1));
        let a2 = device.retrieve::<i32, ()>(100 * 64, (&a2, &a2));

        let a3 = device.retrieve::<i32, ()>(100 * 64, (&a2, &w2));
        let a4 = device.retrieve::<i32, ()>(100 * 64, (&a3, &b2));
        let a4 = device.retrieve::<i32, ()>(100 * 64, (&a4, &a4));

        let a5 = device.retrieve::<i32, ()>(100 * 64, (&a4, &w3));
        let a6 = device.retrieve::<i32, ()>(100 * 64, (&a5, &b3));
        let a6 = device.retrieve::<i32, ()>(100 * 64, (&a6, &a6));

        let a7 = device.retrieve::<i32, ()>(100 * 1, (&a6, &w4));
        let a8 = device.retrieve::<i32, ()>(100 * 1, (&a7, &b4));
        let _loss = device.retrieve::<i32, ()>(100, (&a8, &targets));

        let cts = device.graph().cache_traces();
        assert_eq!(
            cts,
            [
                CacheTrace {
                    cache_idx: 10,
                    use_cache_idx: vec![
                        //   Ident { idx: 0, len: 6400 },
                        Ident { idx: 11, len: 6400 },
                        Ident { idx: 12, len: 6400 },
                        Ident { idx: 13, len: 6400 },
                        Ident { idx: 14, len: 6400 },
                        Ident { idx: 15, len: 6400 },
                        Ident { idx: 16, len: 6400 },
                        Ident { idx: 17, len: 6400 },
                        Ident { idx: 18, len: 6400 }
                    ]
                },
                CacheTrace {
                    cache_idx: 19,
                    use_cache_idx: vec![
                        //   Ident { idx: 0, len: 6400 },
                        Ident { idx: 20, len: 100 },
                        Ident { idx: 21, len: 100 },
                    ]
                }
            ]
        )
    }

    #[test]
    fn test_no_cache_trace_in_graph() {
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        let c = graph.add_node(10, a.idx, b.idx);

        let trace = graph.trace_cache_path(&c);
        graph.cache_traces();

        assert_eq!(Vec::<Node>::new(), trace);
    }

    #[test]
    fn test_multiple_traces() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::new();

        // idx: 0, deps: [] (0)
        let a = graph.add_leaf(10);

        // idx: 1, deps: [0, 0] (1)
        let _b = graph.add_node(10, a.idx, a.idx);

        // idx: 2
        let _z = graph.add_leaf(10);

        // idx: 3
        let _z = graph.add_leaf(10);

        // idx: 4, deps: [0, 1] (0)
        let c = graph.add_node(12, a.idx, a.idx);

        // idx: 5, deps: [2, 2] (1)
        let d = graph.add_node(12, c.idx, c.idx);

        // idx: 6, deps: [3, 1] (2)
        let _e = graph.add_node(12, d.idx, a.idx);

        // idx: 7
        let f = graph.add_node(10, _b.idx, _z.idx);

        // idx: 8
        let _g = graph.add_node(10, f.idx, _z.idx);

        let traces = graph.cache_traces();

        assert_eq!(
            [
                CacheTrace {
                    cache_idx: 1,
                    use_cache_idx: vec![Ident { idx: 7, len: 10 }, Ident { idx: 8, len: 10 }]
                },
                CacheTrace {
                    cache_idx: 4,
                    use_cache_idx: vec![Ident { idx: 5, len: 12 }, Ident { idx: 6, len: 12 }]
                }
            ],
            &*traces
        );
    }
}

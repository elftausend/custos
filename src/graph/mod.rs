#[cfg(not(feature = "no-std"))]
use crate::Ident;

use core::cell::{Ref, RefMut};

#[cfg(feature = "opt-cache")]
use crate::{CacheReturn, DeviceError};

pub use add_graph::*;
pub use node::*;

mod add_graph;
mod node;

#[cfg(not(feature = "no-std"))]
mod graph_struct;

#[cfg(not(feature = "no-std"))]
pub use graph_struct::*;

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

#[cfg(feature = "no-std")]
impl NodeIdx for GlobalCount {}

/// A dummy graph for no-std.
#[cfg(feature = "no-std")]
pub struct Graph<IdxFrom: NodeIdx> {
    _p: core::marker::PhantomData<IdxFrom>,
}

#[cfg(feature = "no-std")]
impl<IdxFrom: NodeIdx> Graph<IdxFrom> {
    /// This function will panic. Disable the `no-std` feature to use this function.
    #[inline]
    pub fn add_leaf(&mut self, _len: usize) -> Node {
        unimplemented!("Not available in no-std mode")
    }

    /// This function will panic. Disable the `no-std` feature to use this function.
    #[inline]
    pub fn add_node(&mut self, _len: usize, _lhs_idx: usize, _rhs_idx: usize) -> Node {
        unimplemented!("Not available in no-std mode")
    }
}

/// A `CacheTrace` is a list of nodes that shows which [`Buffer`](crate::Buffer)s could use the same cache.
#[cfg(not(feature = "no-std"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheTrace {
    /// This identifier is the common cache index / ident. All the other idents in `use_cache_ids` can use this ident to share memory.
    pub cache_id: Ident,
    /// The identifiers of the nodes that can use the common cache entry of `cache_id`.
    pub use_cache_ids: Vec<Ident>,
}

/// Returns a mutable reference to the graph.
pub trait GraphReturn<IdxFrom: NodeIdx = GlobalCount> {
    /// Returns a reference to [`Graph`].
    fn graph(&self) -> Ref<Graph<IdxFrom>>;
    /// Returns a mutable reference to [`Graph`].
    fn graph_mut(&self) -> RefMut<Graph<IdxFrom>>;
}

/// Optimizes [`Graph`] and [`Cache`](crate::Cache) to achive a lower memory footprint.
#[cfg(feature = "opt-cache")]
pub trait GraphOpt {
    /// Optimizes [`Graph`] and [`Cache`](crate::Cache) to achive a lower memory footprint.
    fn optimize(&self) -> crate::Result<()>
    where
        Self: GraphReturn + CacheReturn + crate::PtrConv,
    {
        let mut cache = self.cache_mut();
        for trace in self.graph().cache_traces() {
            for node in &trace.use_cache_ids {
                // insert the common / optimized pointer in all the other nodes
                // this deallocates the old pointers
                let ptr = cache
                    .nodes
                    .get(&trace.cache_id)
                    .ok_or(DeviceError::GraphOptimization)?
                    .clone();
                cache.nodes.insert(*node, ptr);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "opt-cache")]
impl<D: GraphReturn> GraphOpt for D {}

#[cfg(not(feature = "no-std"))]
#[cfg(test)]
mod tests {
    use crate::{set_count, CacheTrace, Graph, Ident, Node, NodeCount};

    #[test]
    fn test_is_leaf() {
        let mut graph = Graph::<NodeCount>::new();
        let node = graph.add_leaf(0);
        assert!(node.is_leaf());

        let node = graph.add_node(10, 1, 2);
        assert!(!node.is_leaf());
    }

    #[test]
    fn test_cache_trace() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::<NodeCount>::new();
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

        let trace = graph.trace_cache_path_raw(&c);
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
        let mut graph = Graph::<NodeCount>::new();
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

        let trace = graph.trace_cache_path_raw(&c);
        assert_eq!(Vec::<Node>::new(), trace);
    }

    #[test]
    fn test_cache_trace_2() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::<NodeCount>::new();
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

        let trace = graph.trace_cache_path_raw(&c);
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
        let mut graph = Graph::<NodeCount>::new();
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

        let trace = graph.trace_cache_path_raw(&c);
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
        let mut graph = Graph::<NodeCount>::new();
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
                cache_id: Ident { idx: 2, len: 10 },
                use_cache_ids: vec![Ident { idx: 3, len: 10 }, Ident { idx: 4, len: 10 },],
            },
            traces[0]
        );
    }

    #[test]
    fn test_leafed_diff_len_trace() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::<NodeCount>::new();
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
                cache_id: Ident { idx: 4, len: 12 },
                use_cache_ids: vec![Ident { idx: 5, len: 12 }, Ident { idx: 6, len: 12 },],
            }],
            &*traces
        );
    }

    #[test]
    fn test_cache_trace_neural_net() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::<NodeCount>::new();
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
                    cache_id: Ident { idx: 10, len: 6400 },
                    use_cache_ids: vec![
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
                    cache_id: Ident { idx: 19, len: 100 },
                    use_cache_ids: vec![
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
        let mut graph = Graph::<NodeCount>::new();
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

        let trace = graph.trace_cache_path_raw(&c);

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
                cache_id: Ident { idx: 2, len: 10 },
                use_cache_ids: vec![Ident { idx: 3, len: 10 }, Ident { idx: 4, len: 10 }]
            }]
        );
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "opt-cache")]
    #[test]
    fn test_from_retrieve_sine_neural_net() {
        use crate::{Buffer, Device, GraphReturn, CPU};

        let device = CPU::<Base>::new();

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
                    cache_id: Ident { idx: 10, len: 6400 },
                    use_cache_ids: vec![
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
                    cache_id: Ident { idx: 19, len: 100 },
                    use_cache_ids: vec![
                        //   Ident { idx: 0, len: 6400 },
                        Ident { idx: 20, len: 100 },
                        Ident { idx: 21, len: 100 },
                    ]
                }
            ]
        )
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "opt-cache")]
    #[test]
    fn test_from_retrieve_sliced_chained_perf_example() {
        use crate::{Buffer, Device, GraphReturn, CPU};

        let device = CPU::<Base>::new();

        // idx: 0, deps: []
        let x: Buffer = device.buffer([1.; 1000]);
        // idx: 1, deps: []
        let b: Buffer = device.buffer([1.1; 1000]);

        // idx: 2, deps: [0, 0]
        let squared = device.retrieve::<f32, ()>(1000, (&x, &x));
        // idx: 3, deps: [1, 0]
        let add = device.retrieve::<f32, ()>(1000, (&b, &x));
        // idx: 4, deps: [3, 1]
        let mul_b = device.retrieve::<f32, ()>(1000, (&add, &b));
        // idx: 5, deps: [2, 0]
        let mul = device.retrieve::<f32, ()>(1000, (&squared, &x));
        // idx: 6, deps: [5, 4]
        let _out = device.retrieve::<f32, ()>(1000, (&mul, &mul_b));

        let traces = device.graph().cache_traces();
        assert_eq!(
            traces,
            vec![
                CacheTrace {
                    cache_id: Ident { idx: 2, len: 1000 },
                    use_cache_ids: vec![Ident { idx: 5, len: 1000 }, Ident { idx: 6, len: 1000 },]
                },
                CacheTrace {
                    cache_id: Ident { idx: 3, len: 1000 },
                    use_cache_ids: vec![Ident { idx: 4, len: 1000 },]
                }
            ]
        );
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "opt-cache")]
    #[test]
    fn test_from_retrieve_sliced_chained_perf_example_optimize_cache() {
        use crate::{Buffer, CacheReturn, Device, GraphOpt, CPU};

        let device = CPU::<Base>::new();

        // idx: 0, deps: []
        let x: Buffer = device.buffer([1.; 1000]);
        // idx: 1, deps: []
        let b: Buffer = device.buffer([1.1; 1000]);

        // idx: 2, deps: [0, 0]
        let squared = device.retrieve::<f32, ()>(1000, (&x, &x));
        // idx: 3, deps: [1, 0]
        let add = device.retrieve::<f32, ()>(1000, (&b, &x));
        // idx: 4, deps: [3, 1]
        let mul_b = device.retrieve::<f32, ()>(1000, (&add, &b));
        // idx: 5, deps: [2, 0]
        let mul = device.retrieve::<f32, ()>(1000, (&squared, &x));
        // idx: 6, deps: [5, 4]
        let out = device.retrieve::<f32, ()>(1000, (&mul, &mul_b));

        device.optimize().unwrap();
        let nodes = device.cache().nodes.clone();

        assert_eq!(nodes.get(&squared.id()), nodes.get(&mul.id()));
        assert_eq!(nodes.get(&squared.id()), nodes.get(&out.id()));

        assert_eq!(nodes.get(&add.id()), nodes.get(&mul_b.id()));
    }

    #[test]
    fn test_no_cache_trace_in_graph() {
        let mut graph = Graph::<NodeCount>::new();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        let c = graph.add_node(10, a.idx, b.idx);

        let trace = graph.trace_cache_path_raw(&c);
        graph.cache_traces();

        assert_eq!(Vec::<Node>::new(), trace);
    }

    #[test]
    fn test_multiple_traces() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::<NodeCount>::new();

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
                    cache_id: Ident { idx: 1, len: 10 },
                    use_cache_ids: vec![Ident { idx: 7, len: 10 }, Ident { idx: 8, len: 10 }]
                },
                CacheTrace {
                    cache_id: Ident { idx: 4, len: 12 },
                    use_cache_ids: vec![Ident { idx: 5, len: 12 }, Ident { idx: 6, len: 12 }]
                }
            ],
            &*traces
        );
    }
}

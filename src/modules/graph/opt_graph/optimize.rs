use super::OptGraph;
use crate::modules::graph::node::Node;
use std::collections::HashSet;

#[derive(Debug, PartialEq, Eq, Default)]
pub struct CacheTrace {
    cache_idx: usize,
    use_cache_idxs: Vec<usize>,
}

type TraceIdx = usize;

impl OptGraph {
    /// Calculates multiple unique [`CacheTrace`]s.
    /// Unique meaning that no two [`CacheTrace`]s share some same [`Node`].
    pub fn cache_traces(&self) -> Vec<CacheTrace> {
        let mut traces = vec![];
        let mut visited_nodes = HashSet::new();

        for node in self.nodes.iter().filter(|node| !node.is_leaf()) {
            if visited_nodes.contains(&node.idx) {
                continue;
            }

            let trace = self.trace_cache_path_raw(node);

            if trace.is_empty() {
                continue;
            }

            traces.push(CacheTrace {
                cache_idx: node.idx,
                use_cache_idxs: trace
                    .into_iter()
                    .filter_map(|node_idx| {
                        if visited_nodes.contains(&node_idx) {
                            return None;
                        }
                        visited_nodes.insert(node_idx);
                        Some(node_idx)
                    })
                    .collect(),
            });
        }

        traces
    }

    /// Calculates the cache trace for a starting node.
    /// A cache trace is a list of nodes that shows which [`Buffer`](crate::Buffer)s could use the same cache.
    pub fn trace_cache_path_raw(&self, trace_at: &Node) -> Vec<TraceIdx> {
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
            trace.push(idx);

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
    /// use custos::OptGraph;
    ///
    /// let mut graph = OptGraph::default();
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

#[cfg(not(feature = "no-std"))]
#[cfg(test)]
mod tests {
    use crate::modules::graph::opt_graph::{optimize::CacheTrace, OptGraph};

    #[test]
    fn test_cache_trace() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1]
        let c = graph.add_node(10, vec![a, b]);

        // idx: 3, deps: [2, 2]
        let d = graph.add_node(10, vec![c, c]);

        // idx: 4, deps: [3, 1]
        let _e = graph.add_node(10, vec![d, b]);

        // idx: 5, deps: [2, 1]
        //let f = graph.add_node(10, vec![c, b]);

        let trace = graph.trace_cache_path_raw(graph.node(c));
        assert_eq!(vec![3, 4], trace);

        let _traces = graph.cache_traces();
        //println!("traces: {traces:?}");
    }

    #[test]
    fn test_no_cache_trace() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1]
        let c = graph.add_node(10, vec![a, b]);

        // idx: 3, deps: [2, 2]
        let d = graph.add_node(10, vec![c, c]);

        // idx: 4, deps: [3, 1]
        let _e = graph.add_node(10, vec![d, b]);

        // idx: 5, deps: [2, 1]
        let _f = graph.add_node(10, vec![c, b]);

        let trace = graph.trace_cache_path_raw(graph.node(c));
        assert_eq!(Vec::<usize>::new(), trace);
    }

    #[test]
    fn test_cache_trace_2() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();
        let a = graph.add_leaf(10); // idx: 0
        let b = graph.add_leaf(10); // idx: 1
        let u = graph.add_leaf(10); // idx: 2

        // idx: 3, deps: [0, 1]
        let c = graph.add_node(10, vec![a, b]);

        // idx: 4, deps: [0, 2]
        let _z = graph.add_node(10, vec![a, u]);

        // idx: 5, deps: [3, 3]
        let d = graph.add_node(10, vec![c, c]);

        // idx: 6, deps: [5, 1]
        let _e = graph.add_node(10, vec![d, b]);

        let trace = graph.trace_cache_path_raw(graph.node(c));
        assert_eq!(vec![5, 6], trace);
    }

    #[test]
    fn test_cache_trace_break_not_anymore() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1]
        let c = graph.add_node(10, vec![a, b]);

        // idx: 3, deps: [2, 2]
        let d = graph.add_node(10, vec![c, c]);

        // idx: 4, deps: [3, 0]
        let _u = graph.add_node(10, vec![d, a]);

        // idx: 5, deps: [3, 1]
        let _e = graph.add_node(10, vec![d, b]);

        println!("traces: {:?}", graph.cache_traces());

        let trace = graph.trace_cache_path_raw(graph.node(c));
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

        assert!(graph.is_path_optimizable(graph.node(c)));
        assert!(!graph.is_path_optimizable(graph.node(d)));
    }

    #[test]
    fn test_trace_all() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1] (0)
        let c = graph.add_node(10, vec![a, b]);

        // idx: 3, deps: [2, 2] (1)
        let d = graph.add_node(10, vec![c, c]);

        // idx: 4, deps: [3, 1] (2)
        let _e = graph.add_node(10, vec![d, b]);

        let traces = graph.cache_traces();

        assert_eq!(traces.len(), 1);

        assert_eq!(
            CacheTrace {
                cache_idx: 2,
                use_cache_idxs: vec![3, 4]
            },
            traces[0]
        );
    }

    #[test]
    fn test_leafed_diff_len_trace() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();
        let a = graph.add_leaf(10);

        let _b = graph.add_node(10, vec![a, a]);

        let _z = graph.add_leaf(10);

        let _z = graph.add_leaf(10);

        // idx: 2, deps: [0, 1] (0)
        let c = graph.add_node(10, vec![a, a]);

        // idx: 3, deps: [2, 2] (1)
        let d = graph.add_node(10, vec![c, c]);

        // idx: 4, deps: [3, 1] (2)
        let _e = graph.add_node(10, vec![d, a]);

        let traces = graph.cache_traces();
        assert_eq!(
            CacheTrace {
                cache_idx: 4,
                use_cache_idxs: vec![5, 6]
            },
            traces[0]
        );
    }

    #[test]
    fn test_cache_trace_neural_net() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();
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

        let a1 = graph.add_node(100 * 64, vec![inputs, w1]);
        let a2 = graph.add_node(100 * 64, vec![a1, b1]);
        let a2 = graph.add_node(100 * 64, vec![a2, a2]);

        let a3 = graph.add_node(100 * 64, vec![a2, w2]);
        let a4 = graph.add_node(100 * 64, vec![a3, b2]);
        let a4 = graph.add_node(100 * 64, vec![a4, a4]);

        let a5 = graph.add_node(100 * 64, vec![a4, w3]);
        let a6 = graph.add_node(100 * 64, vec![a5, b3]);
        let a6 = graph.add_node(100 * 64, vec![a6, a6]);
        let a7 = graph.add_node(100, vec![a6, w4]);
        let a8 = graph.add_node(100, vec![a7, b4]);

        let _loss = graph.add_node(100, vec![a8, targets]);

        let traces = graph.cache_traces();
        assert_eq!(
            traces,
            [
                CacheTrace {
                    cache_idx: 10,
                    use_cache_idxs: vec![11, 12, 13, 14, 15, 16, 17, 18]
                },
                CacheTrace {
                    cache_idx: 19,
                    use_cache_idxs: vec![20, 21]
                }
            ]
        )

        // graph.add_node(10*10, gemm, gemm);
        // bump_count();
    }

    #[test]
    fn test_cache_trace_d() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        // idx: 2, deps: [0, 1]
        let c = graph.add_node(10, vec![a, b]);

        // idx: 3, deps: [2, 2]
        let d = graph.add_node(10, vec![c, c]);

        // idx: 4, deps: [3, 0]
        let _u = graph.add_node(10, vec![a, d]);

        // idx: 5, deps: [3, 1]
        //let _e = graph.add_node(10, vec![d, b]);

        let trace = graph.trace_cache_path_raw(graph.node(c));

        assert_eq!(vec![3, 4], trace);

        assert!(graph.is_path_optimizable(graph.node(c)));
        assert!(graph.is_path_optimizable(graph.node(d)));

        let trace = graph.cache_traces();
        assert_eq!(
            trace,
            [CacheTrace {
                cache_idx: 2,
                use_cache_idxs: vec![3, 4]
            }]
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_from_retrieve_sine_neural_net() {
        use crate::{Base, Buffer, Graph, Retriever, CPU};

        let device = CPU::<Graph<Base>>::new();

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

        let a1: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&inputs, &w1));
        let a2: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&a1, &b1));
        let a2: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&a2, &a2));

        let a3: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&a2, &w2));
        let a4: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&a3, &b2));
        let a4: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&a4, &a4));
        let a5: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&a4, &w3));
        let a6: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&a5, &b3));
        let a6: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 64, (&a6, &a6));
        let a7: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 1, (&a6, &w4));
        let a8: Buffer<i32, _> = device.retrieve::<(), 2>(100 * 1, (&a7, &b4));
        let _loss: Buffer<i32, _> = device.retrieve::<(), 2>(100, (&a8, &targets));

        let cts = device
            .modules
            .graph_trans
            .borrow_mut()
            .opt_graph
            .cache_traces();
        assert_eq!(
            cts,
            [
                CacheTrace {
                    cache_idx: 10,
                    use_cache_idxs: vec![11, 12, 13, 14, 15, 16, 17, 18]
                },
                CacheTrace {
                    cache_idx: 19,
                    use_cache_idxs: vec![20, 21]
                }
            ]
        )
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_from_retrieve_sliced_chained_perf_example() {
        use crate::{Base, Buffer, Device, Graph, Retriever, CPU};

        let device = CPU::<Graph<Base>>::new();

        // idx: 0, deps: []
        let x: Buffer<f32, _> = device.buffer([1.; 1000]);
        // idx: 1, deps: []
        let b: Buffer<f32, _> = device.buffer([1.1; 1000]);

        // idx: 2, deps: [0, 0]
        let squared: Buffer<f32, _> = device.retrieve::<(), 2>(1000, (&x, &x));
        // idx: 3, deps: [1, 0]
        let add: Buffer<f32, _> = device.retrieve::<(), 2>(1000, (&b, &x));
        // idx: 4, deps: [3, 1]
        let mul_b: Buffer<f32, _> = device.retrieve::<(), 2>(1000, (&add, &b));
        // idx: 5, deps: [2, 0]
        let mul: Buffer<f32, _> = device.retrieve::<(), 2>(1000, (&squared, &x));
        // idx: 6, deps: [5, 4]
        let _out: Buffer<f32, _> = device.retrieve::<(), 2>(1000, (&mul, &mul_b));

        let traces = device
            .modules
            .graph_trans
            .borrow_mut()
            .opt_graph
            .cache_traces();

        assert_eq!(
            traces,
            vec![
                CacheTrace {
                    cache_idx: 2,
                    use_cache_idxs: vec![5, 6]
                },
                CacheTrace {
                    cache_idx: 3,
                    use_cache_idxs: vec![4]
                }
            ]
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_from_retrieve_sliced_chained_perf_example_optimize_cache() {
        /*
        use crate::{Buffer, CacheReturn, Device, GraphOpt, CPU};

        let device = CPU::<Base>::new();

        // idx: 0, deps: []
        let x: Buffer = device.buffer([1.; 1000]);
        // idx: 1, deps: []
        let b: Buffer = device.buffer([1.1; 1000]);

        // idx: 2, deps: [0, 0]
        let squared = device.retrieve::<(), 2>(1000, (&x, &x));
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
        */
    }

    #[test]
    fn test_no_cache_trace_in_graph() {
        let mut graph = OptGraph::default();
        let a = graph.add_leaf(10);
        let b = graph.add_leaf(10);

        let c = graph.add_node(10, vec![a, b]);

        let trace = graph.trace_cache_path_raw(graph.node(c));
        graph.cache_traces();

        assert_eq!(Vec::<usize>::new(), trace);
    }

    #[test]
    fn test_multiple_traces() {
        // for: cargo test -- --test-threads=1
        let mut graph = OptGraph::default();

        // idx: 0, deps: [] (0)
        let a = graph.add_leaf(10);

        // idx: 1, deps: [0, 0] (1)
        let _b = graph.add_node(10, vec![a, a]);

        // idx: 2
        let _z = graph.add_leaf(10);

        // idx: 3
        let _z = graph.add_leaf(10);

        // idx: 4, deps: [0, 1] (0)
        let c = graph.add_node(10, vec![a, a]);

        // idx: 5, deps: [2, 2] (1)
        let d = graph.add_node(10, vec![c, c]);

        // idx: 6, deps: [3, 1] (2)
        let _e = graph.add_node(10, vec![d, a]);

        // idx: 7
        let f = graph.add_node(10, vec![_b, _z]);

        // idx: 8
        let _g = graph.add_node(10, vec![f, _z]);

        let traces = graph.cache_traces();

        assert_eq!(
            [
                CacheTrace {
                    cache_idx: 1,
                    use_cache_idxs: vec![7, 8]
                },
                CacheTrace {
                    cache_idx: 4,
                    use_cache_idxs: vec![5, 6]
                }
            ],
            &*traces
        );
    }
}

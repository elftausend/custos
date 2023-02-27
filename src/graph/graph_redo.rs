use std::collections::HashSet;

use crate::{CacheTrace, Graph, Ident, Node};

impl Graph {
    pub fn cache_traces_2(&self) -> Vec<CacheTrace> {
        let mut traces = vec![];
        let mut visited_nodes = HashSet::new();

        for node in self.nodes.iter().filter(|node| !node.is_leaf()) {
            if visited_nodes.contains(node) {
                continue;
            }

            let trace = self.trace_cache_path_2(node);

            if trace.is_empty() {
                continue;
            }

            traces.push(CacheTrace {
                cache_idx: node.idx,
                use_cache_idx: trace
                    .iter()
                    //.filter(|node| !visited_nodes.contains(*node))
                    .map(|node| {
                        visited_nodes.insert(*node);
                        Ident {
                            idx: node.ident_idx,
                            len: node.len,
                        }
                    })
                    .collect(),
            });
        }

        traces
    }

    pub fn trace_cache_path_2(&self, trace_at: &Node) -> Vec<Node> {
        let mut trace = vec![];

        let mut idx = trace_at.idx;

        for check in self.nodes.iter().skip(trace_at.idx + 1) {
            if trace_at.len != check.len || !self.is_path_optimizable(check) {
                continue;
            }

            if check.deps.contains(&idx) {
                idx = check.idx;
                trace.push(*check);
            }
        }

        trace
    }
}

#[cfg(test)]
mod tests {
    use crate::{bump_count, set_count, CacheTrace, Graph, Ident};

    #[test]
    fn test_new_cache_traces() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        bump_count();
        let b = graph.add_leaf(10);
        bump_count();

        // idx: 2, deps: [0, 1] (0)
        let c = graph.add_node(10, a.idx, b.idx);
        bump_count();

        // idx: 3, deps: [2, 2] (1)
        let d = graph.add_node(10, c.idx, c.idx);
        bump_count();

        // idx: 4, deps: [3, 1] (2)
        let _e = graph.add_node(10, d.idx, b.idx);
        bump_count();

        let traces = graph.cache_traces_2();

        assert_eq!(
            CacheTrace {
                cache_idx: 2,
                use_cache_idx: vec![
                    //Ident { idx: 2, len: 10 },
                    Ident { idx: 3, len: 10 },
                    Ident { idx: 4, len: 10 },
                ],
            },
            traces[0]
        );
    }

    #[test]
    fn test_leafed_diff_len_trace2() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::new();
        let a = graph.add_leaf(10);
        bump_count();
        let _b = graph.add_node(10, a.idx, a.idx);
        bump_count();

        let _z = graph.add_leaf(10);
        bump_count();

        let _z = graph.add_leaf(10);
        bump_count();

        // idx: 2, deps: [0, 1] (0)
        let c = graph.add_node(12, a.idx, a.idx);
        bump_count();

        // idx: 3, deps: [2, 2] (1)
        let d = graph.add_node(12, c.idx, c.idx);
        bump_count();

        // idx: 4, deps: [3, 1] (2)
        let _e = graph.add_node(12, d.idx, a.idx);

        let traces = graph.cache_traces_2();

        assert_eq!(
            [CacheTrace {
                cache_idx: 4,
                use_cache_idx: vec![Ident { idx: 5, len: 12 }, Ident { idx: 6, len: 12 },],
            }],
            &*traces
        );
        println!("traces: {traces:?}");
    }

    #[test]
    fn test_multiple_traces() {
        // for: cargo test -- --test-threads=1
        unsafe { set_count(0) };
        let mut graph = Graph::new();

        // idx: 0, deps: [] (0)
        let a = graph.add_leaf(10);
        bump_count();

        // idx: 1, deps: [0, 0] (1)
        let _b = graph.add_node(10, a.idx, a.idx);
        bump_count();

        // idx: 2
        let _z = graph.add_leaf(10);
        bump_count();

        // idx: 3
        let _z = graph.add_leaf(10);
        bump_count();

        // idx: 4, deps: [0, 1] (0)
        let c = graph.add_node(12, a.idx, a.idx);
        bump_count();

        // idx: 5, deps: [2, 2] (1)
        let d = graph.add_node(12, c.idx, c.idx);
        bump_count();

        // idx: 6, deps: [3, 1] (2)
        let _e = graph.add_node(12, d.idx, a.idx);
        bump_count();

        // idx: 7
        let f = graph.add_node(10, _b.idx, _z.idx);
        bump_count();
        // idx: 8
        let _g = graph.add_node(10, f.idx, _z.idx);
        bump_count();

        let traces = graph.cache_traces_2();

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
        println!("traces: {traces:?}");
    }
}

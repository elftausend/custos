use std::collections::HashSet;

use crate::{AddGraph, CacheTrace, Ident, Node, COUNT};

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
        let idx = self.nodes.len();
        let node = COUNT.with(|count| {
            Node {
                // subtracting 1, because the count is increased beforehand.
                //ident_idx: idx,
                idx,
                deps: [idx, idx],
                len,
            }
        });
        self.nodes.push(node);
        node
    }

    pub fn add_node(&mut self, len: usize, lhs_idx: usize, rhs_idx: usize) -> Node {
        let idx = self.nodes.len();
        let node = COUNT.with(|count| {
            Node {
                // subtracting 1, because the count is increased beforehand.
                // ident_idx: idx,
                idx,
                deps: [lhs_idx, rhs_idx],
                len,
            }
        });
        self.nodes.push(node);
        node
    }

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
                            idx: node.idx,
                            len: node.len,
                        }
                    })
                    .collect(),
            });
        }

        traces
    }

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

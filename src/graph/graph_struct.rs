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
                        len: node.len,
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

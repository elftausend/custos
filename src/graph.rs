use std::cell::RefMut;

pub trait GraphReturn {
    fn graph(&self) -> RefMut<Graph>;
}

#[derive(Default, Debug)]
pub struct Graph {
    nodes: Vec<GNode>,
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_leaf(&mut self, len: usize) -> GNode {
        let idx = self.nodes.len();
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

    pub fn trace_cache_path(&self, trace_at: &GNode) -> Option<Vec<GNode>> {
        if !self.is_path_optimizable(trace_at) {
            return None;
        }

        let mut trace = vec![*trace_at];

        let mut idx = trace_at.idx;
        for check in &self.nodes[trace_at.idx + 1..] {
            if !self.is_path_optimizable(check) {
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
            if !check.deps.contains(&check_at.idx) {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

#[cfg(test)]
mod tests {
    use crate::{GNode, Graph};

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
        let u = graph.add_node(10, d.idx, a.idx);

        // idx: 5, deps: [3, 1]
        let e = graph.add_node(10, d.idx, b.idx);


        let trace = graph.trace_cache_path(&c);

        assert!(graph.is_path_optimizable(&c));
        assert!(!graph.is_path_optimizable(&d));
        println!("trace: {trace:?}");
    }
}

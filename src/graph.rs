
#[derive(Default, Debug)]
pub struct Graph {
    nodes: Vec<GNode>
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new()
        }
    }

    pub fn add_leaf(&mut self) {
        let idx = self.nodes.len();
        self.nodes.push(GNode { idx, deps: [idx, idx] })
    }

    pub fn add_node(&mut self, lhs_idx: usize, rhs_idx: usize) -> GNode {
        let idx = self.nodes.len();
        let node = GNode { idx, deps: [lhs_idx, rhs_idx] };
        self.nodes.push(node);
        node
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GNode {
    pub idx: usize,
    pub deps: [usize; 2]
}
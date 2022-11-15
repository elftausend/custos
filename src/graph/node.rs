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

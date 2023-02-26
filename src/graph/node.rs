#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node {
    pub ident_idx: usize,
    pub idx: usize,
    pub deps: [usize; 2],
    pub len: usize,
}



impl Node {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.idx == self.deps[0] && self.idx == self.deps[1]
    }
}

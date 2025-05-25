/// A node in the [`Graph`](crate::Graph).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node {
    /// The index of the node.
    pub idx: usize,
    /// The indices of the nodes that are dependencies of this node.
    pub deps: Vec<usize>,
    /// The amount of elements a corresponding [`Buffer`](crate::Buffer) has.
    pub len: usize,
    pub skip: bool,
}

impl Node {
    /// `true` if the node is a leaf.
    /// # Example
    /// ```
    /// use custos::Node;
    ///
    /// let node = Node {
    ///     idx: 0,
    ///     deps: vec![0, 0],
    ///     len: 10,
    /// };
    ///
    /// assert!(node.is_leaf());
    ///
    /// let node = Node {
    ///     idx: 1,
    ///     deps: vec![0, 0],
    ///     len: 10,
    /// };
    /// assert!(!node.is_leaf());
    /// ```
    #[inline]
    pub fn is_leaf(&self) -> bool {
        if self.deps.is_empty() {
            return true;
        }
        self.deps.iter().all(|dep| *dep == self.idx)
    }
}

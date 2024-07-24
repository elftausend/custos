use crate::{Buffers, Operation};

pub struct ExecIter<'b, B, T> {
    pub(super) operations: std::slice::Iter<'b, Operation<B, T>>,
    pub(super) buffers: &'b mut Buffers<B>,
}

impl<'b, B, T> Iterator for ExecIter<'b, B, T> {
    type Item = crate::Result<()>;

    fn next(&mut self) -> Option<Self::Item> {
        let op = self.operations.next()?;
        Some((op.op)(self.buffers))
    }
}

impl<'b, B, T> DoubleEndedIterator for ExecIter<'b, B, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let op = self.operations.next_back()?;
        Some((op.op)(self.buffers))
    }
}

impl<'b, B, T> ExactSizeIterator for ExecIter<'b, B, T> {
    fn len(&self) -> usize {
        self.operations.len()
    }
}

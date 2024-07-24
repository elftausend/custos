use crate::{Buffers, Operation2};

pub struct ExecIter<'a, 'b, B, T> {
    pub(super) operations: std::slice::Iter<'b, Operation2<'a, B, T>>,
    pub(super) buffers: &'b mut Buffers<B>,
}

impl<'a, 'b, B, T> Iterator for ExecIter<'a, 'b, B, T> {
    type Item = crate::Result<()>;

    fn next(&mut self) -> Option<Self::Item> {
        let op = self.operations.next()?;
        Some((op.op)(self.buffers))
    }
}

impl<'a, 'b, B, T> DoubleEndedIterator for ExecIter<'a, 'b, B, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let op = self.operations.next_back()?;
        Some((op.op)(self.buffers))
    }
}

impl<'a, 'b, B, T> ExactSizeIterator for ExecIter<'a, 'b, B, T> {
    fn len(&self) -> usize {
        self.operations.len()
    }
}

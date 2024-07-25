use crate::{Buffers, Device, Operation};

pub struct ExecIter<'_6, B, T, D> {
    pub(super) operations: std::slice::Iter<'_6, Operation<B, T>>,
    pub(super) buffers: &'_6 mut Buffers<B>,
    pub(super) device: &'_6 D,
}

impl<'b, B, T, D: Device + 'static> Iterator for ExecIter<'b, B, T, D> {
    type Item = crate::Result<()>;

    fn next(&mut self) -> Option<Self::Item> {
        let op = self.operations.next()?;
        Some((op.op)(self.buffers, self.device))
    }
}

impl<'b, B, T, D: Device + 'static> DoubleEndedIterator for ExecIter<'b, B, T, D> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let op = self.operations.next_back()?;
        Some((op.op)(self.buffers, self.device))
    }
}

impl<'b, B, T, D: Device + 'static> ExactSizeIterator for ExecIter<'b, B, T, D> {
    fn len(&self) -> usize {
        self.operations.len()
    }
}

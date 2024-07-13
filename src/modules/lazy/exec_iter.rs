use core::marker::PhantomData;

use crate::{BorrowCache, Buffers, Operation2, UniqueId, UpdateArgsDynable};

use super::lazy_graph::Operation;

pub struct ExecIter<'a, B, T> {
    pub(super) operations: std::slice::IterMut<'a, Operation<B, T>>,
    pub(super) buffers: &'a mut Buffers<B>,
}

pub fn exec_op<B>(
    args: &mut Box<dyn UpdateArgsDynable<B>>,
    op: &fn(*mut ()) -> crate::Result<()>,
    ids_to_check: &[Option<UniqueId>],
    buffers: &mut Buffers<B>,
) -> crate::Result<()> {
    args.update_args_dynable(ids_to_check, buffers)?;
    let args = core::ptr::addr_of_mut!(**args) as *mut ();
    op(args)
}

impl<'a, B, T> Iterator for ExecIter<'a, B, T> {
    type Item = crate::Result<()>;

    fn next(&mut self) -> Option<Self::Item> {
        let op = self.operations.next()?;
        Some(exec_op(&mut op.args, &op.op, &op.arg_ids, self.buffers))
    }
}

impl<'a, B, T> DoubleEndedIterator for ExecIter<'a, B, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let op = self.operations.next_back()?;
        Some(exec_op(&mut op.args, &op.op, &op.arg_ids, self.buffers))
    }
}

impl<'a, B, T> ExactSizeIterator for ExecIter<'a, B, T> {
    fn len(&self) -> usize {
        self.operations.len()
    }
}

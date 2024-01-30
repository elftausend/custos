use crate::{Buffers, UniqueId, UpdateArgs, UpdateArgsDynable};

pub struct ExecIter<'a, B> {
    pub(super) ids_to_check: std::slice::Iter<'a, Vec<Option<UniqueId>>>,
    pub(super) ops: std::slice::Iter<'a, fn(*mut ()) -> crate::Result<()>>,
    pub(super) args: std::slice::IterMut<'a, Box<dyn UpdateArgsDynable<B>>>,
    pub(super) buffers: &'a mut Buffers<B>,
}

pub fn exec_op<B>(
    args: &mut Box<dyn UpdateArgsDynable<B>>,
    op: &fn(*mut ()) -> crate::Result<()>,
    ids_to_check: &[Option<UniqueId>],
    buffers: &mut Buffers<B>,
) -> crate::Result<()> {
    args.update_args_dynable(ids_to_check, buffers)?;

    let args = &mut **args as *mut _ as *mut ();
    op(args)
}

impl<'a, B> Iterator for ExecIter<'a, B> {
    type Item = crate::Result<()>;

    fn next(&mut self) -> Option<Self::Item> {
        let ids_to_check = self.ids_to_check.next()?;
        let op = self.ops.next()?;
        let args = self.args.next()?;
        Some(exec_op(args, op, ids_to_check, self.buffers))
    }
}

impl<'a, B> DoubleEndedIterator for ExecIter<'a, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ids_to_check = self.ids_to_check.next_back()?;
        let op = self.ops.next_back()?;
        let args = self.args.next_back()?;
        Some(exec_op(args, op, ids_to_check, self.buffers))
    }
}

impl<'a, B> ExactSizeIterator for ExecIter<'a, B> {
    fn len(&self) -> usize {
        self.ids_to_check.len()
    }
}

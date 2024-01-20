use crate::{Buffers, UniqueId, UpdateArgs};

pub struct ExecIter<'a> {
    pub(super) ids_to_check: std::slice::Iter<'a, Vec<Option<UniqueId>>>,
    pub(super) ops: std::slice::Iter<'a, fn(*mut ()) -> crate::Result<()>>,
    pub(super) args: std::slice::IterMut<'a, Box<dyn UpdateArgs>>,
    pub(super) buffers: &'a mut Buffers,
}

pub fn exec_op(
    args: &mut Box<dyn UpdateArgs>,
    op: &fn(*mut ()) -> crate::Result<()>,
    ids_to_check: &[Option<UniqueId>],
    buffers: &mut Buffers,
) -> crate::Result<()> {
    args.update_args(ids_to_check, buffers)?;

    let args = &mut **args as *mut _ as *mut ();
    op(args)
}

impl<'a> Iterator for ExecIter<'a> {
    type Item = crate::Result<()>;

    fn next(&mut self) -> Option<Self::Item> {
        let ids_to_check = self.ids_to_check.next()?;
        let op = self.ops.next()?;
        let args = self.args.next()?;
        Some(exec_op(args, op, ids_to_check, self.buffers))
    }
}

impl<'a> DoubleEndedIterator for ExecIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ids_to_check = self.ids_to_check.next_back()?;
        let op = self.ops.next_back()?;
        let args = self.args.next_back()?;
        Some(exec_op(args, op, ids_to_check, self.buffers))
    }
}

impl<'a> ExactSizeIterator for ExecIter<'a> {
    fn len(&self) -> usize {
        self.ids_to_check.len()
    }
}

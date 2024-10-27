use core::{mem::ManuallyDrop, ops::{Deref, DerefMut}};

use crate::{CowMutCell, HasId, PtrType, ShallowCopy};

#[derive(Debug)]
pub struct Guard<'a, T> {
    data: CowMutCell<'a, T>,
}

impl<'a, T> Guard<'a, T> {
    #[inline]
    pub fn new(data: CowMutCell<'a, T>) -> Self {
        Self { data }
    }

    pub fn map<F, U>(self, f: F) -> Guard<'a, U>
    where
        F: FnOnce(CowMutCell<'a, T>) -> CowMutCell<'a, U>,
    {
        let Guard { data } = self;
        Guard {
            data: f(data),
        }
    }
}

impl<'a, T> Deref for Guard<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T> DerefMut for Guard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, T: PtrType> PtrType for Guard<'a, T> {
    #[inline]
    fn size(&self) -> usize {
        self.data.size()
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        self.data.flag()
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        self.data.set_flag(flag);
    }
}

impl<'a, T: HasId> HasId for Guard<'a, T> {
    #[inline]
    fn id(&self) -> crate::Id {
        self.data.id()
    }
}

impl<'a, T> ShallowCopy for Guard<'a, T> {
    unsafe fn shallow(&self) -> Self {
        todo!()
    }
}


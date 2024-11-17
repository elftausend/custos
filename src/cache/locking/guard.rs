use core::ops::{Deref, DerefMut};

use crate::{CowMutCell, HasId, HostPtr, PtrType, ShallowCopy, ToDim};

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
        Guard { data: f(data) }
    }

    #[inline]
    pub fn make_static(self) -> Option<Guard<'static, T>> {
        match self.data {
            CowMutCell::Borrowed(_) => None,
            CowMutCell::BorrowedMut(_) => None,
            CowMutCell::Owned(data) => Some(Guard::new(CowMutCell::Owned(data))),
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

impl<'a, T, P: PtrType + HostPtr<T>> HostPtr<T> for Guard<'a, P> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.data.get_ref().ptr()
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.data.get_mut().unwrap().ptr_mut()
    }
}

impl<'a, P> ToDim for Guard<'a, P> {
    type Out = Self;

    #[inline]
    fn to_dim(self) -> Self::Out {
        self
    }

    #[inline]
    fn as_dim(&self) -> &Self::Out {
        self
    }
}

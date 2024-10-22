use core::ops::{Deref, DerefMut};
use core::cell::{Ref, RefMut};

pub type CowMutCell<'a, T> = CowMut<T, RefMut<'a, T>, Ref<'a, T>>;
pub type CowMutRef<'a, T> = CowMut<T, &'a T, &'a mut T>;

#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CowMut<T, M, R> {
    Borrowed(R),
    BorrowedMut(M),
    Owned(T),
}

impl<T: Default, M, R> Default for CowMut<T, M, R> {
    fn default() -> Self {
        CowMut::Owned(T::default())
    }
}

impl<T, M: DerefMut<Target = T>, R: Deref<Target = T>> CowMut<T, M, R> {
    #[inline]
    pub fn get_ref(&self) -> &T {
        match self {
            CowMut::Borrowed(borrowed) => borrowed,
            CowMut::BorrowedMut(borrowed_mut) => borrowed_mut,
            CowMut::Owned(owned) => owned,
        }
    }

    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        match self {
            CowMut::Borrowed(_borrowed) => None,
            CowMut::BorrowedMut(borrowed_mut) => Some(borrowed_mut),
            CowMut::Owned(owned) => Some(owned),
        }
    }

    #[inline]
    pub fn is_owned(&self) -> bool {
        match self {
            CowMut::Borrowed(_) | CowMut::BorrowedMut(_) => false,
            CowMut::Owned(_) => true,
        }
    }

    #[inline]
    pub fn into_owned(self) -> T
    where
        T: Clone,
    {
        match self {
            CowMut::Borrowed(b) => b.clone(),
            CowMut::BorrowedMut(b) => b.clone(),
            CowMut::Owned(o) => o,
        }
    }
}

impl<T, M: DerefMut<Target = T>, R: Deref<Target = T>> Deref for CowMut<T, M, R> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            CowMut::Borrowed(b) => b,
            CowMut::BorrowedMut(b) => b,
            CowMut::Owned(o) => o,
        }
    }
}

impl<T, M: DerefMut<Target = T>, R: Deref<Target = T>> DerefMut for CowMut<T, M, R> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
            .expect("Cannot get mutable reference from immutable data")
    }
}

use core::ops::{Deref, DerefMut};

#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug)]
pub enum CowMut<'a, T> {
    BorrowedMut(&'a mut T),
    Owned(T),
}

impl<T: Default> Default for CowMut<'_, T> {
    fn default() -> Self {
        CowMut::Owned(T::default())
    }
}

impl<'a, T> CowMut<'a, T> {
    #[inline]
    pub fn get_ref(&self) -> &T {
        match self {
            // CowMut::Borrowed(borrowed) => borrowed,
            CowMut::BorrowedMut(borrowed_mut) => borrowed_mut,
            CowMut::Owned(owned) => owned,
        }
    }

    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        match self {
            // CowMut::Borrowed(_borrowed) => None,
            CowMut::BorrowedMut(borrowed_mut) => Some(borrowed_mut),
            CowMut::Owned(owned) => Some(owned),
        }
    }

    #[inline]
    pub fn is_owned(&self) -> bool {
        match self {
            CowMut::BorrowedMut(_) => false,
            CowMut::Owned(_) => true
        }
    }

    #[inline]
    pub fn into_owned(self) -> T 
    where 
        T: Clone
    {
        match self {
            CowMut::BorrowedMut(b) => b.clone(),
            CowMut::Owned(o) => o,
        }
    }
}

impl<'a, T> Deref for CowMut<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            CowMut::BorrowedMut(b) => b,
            CowMut::Owned(o) => o,
        }
    }
}

impl<'a, T> DerefMut for CowMut<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            CowMut::BorrowedMut(b) => b,
            CowMut::Owned(o) => o,
        }
    }
}

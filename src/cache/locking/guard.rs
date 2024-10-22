use core::{mem::ManuallyDrop, ops::{Deref, DerefMut}};

use crate::CowMutCell;

#[derive(Debug)]
pub struct Guard<'a, T> {
    data: Option<CowMutCell<'a, T>>,
}

impl<'a, T> Guard<'a, T> {
    #[inline]
    pub fn new(data: Option<CowMutCell<'a, T>>) -> Self {
        Self { data }
    }

    pub fn map<F, U>(self, f: F) -> Guard<'a, U>
    where
        F: FnOnce(CowMutCell<'a, T>) -> CowMutCell<'a, U>,
    {
        let mut guard = ManuallyDrop::new(self);
        let data = core::mem::take(&mut guard.data);

        Guard {
            data: data.map(|x| f(x)),
        }
    }
}

impl<'a, T> Deref for Guard<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for Guard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut().unwrap()
    }
}

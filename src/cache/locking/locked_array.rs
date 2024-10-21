use std::cell::{RefCell, RefMut};

use crate::cow_mut::CowMutCell;

use super::{Guard, LockInfo, State};

pub struct LockedArray2<T: Sized, const N: usize = 1000> {
    data: [RefCell<Option<T>>; N],
}

impl<T, const N: usize> Default for LockedArray2<T, N> {
    #[inline]
    fn default() -> Self {
        Self {
            data: [const { RefCell::new(None) }; N],
        }
    }
}

impl<T, const N: usize> LockedArray2<T, N> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, const N: usize> LockedArray2<T, N> {
    pub fn set(&self, id: usize, data: T) {
        assert!(self.data[id].borrow().is_none());
        *self.data[id].borrow_mut() = Some(data);
    }

    pub fn get<'a>(&'a self, id: usize) -> State<Guard<'a, T>> {
        match self.data[id].try_borrow_mut() {
            Ok(data) => {
                if data.is_none() {
                    return State::Err(LockInfo::None);
                }
                return State::Ok(Guard::new(Some(CowMutCell::BorrowedMut(RefMut::map(
                    data,
                    |data| data.as_mut().unwrap(),
                )))));
            }
            Err(_) => return State::Err(LockInfo::Locked),
        }
    }
}

#[cfg(test)]
mod tests {
}

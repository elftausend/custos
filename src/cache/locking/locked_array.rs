use core::cell::{Ref, RefCell, RefMut};

use crate::cow_mut::CowMutCell;

use super::{Guard, LockInfo, State};

pub struct LockedArray<T: Sized, const N: usize = 1000> {
    data: [RefCell<Option<T>>; N],
}

impl<T, const N: usize> Default for LockedArray<T, N> {
    #[inline]
    fn default() -> Self {
        Self {
            data: [const { RefCell::new(None) }; N],
        }
    }
}

impl<T, const N: usize> LockedArray<T, N> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, const N: usize> LockedArray<T, N> {
    pub fn set(&self, id: usize, data: T) {
        // not required to check this
        assert!(self.data[id].borrow().is_none());
        *self.data[id].borrow_mut() = Some(data);
    }

    pub fn get<'a>(&'a self, id: usize) -> State<Guard<'a, T>> {
        match self.data[id].try_borrow() {
            Ok(data) => {
                if data.is_none() {
                    return State::Err(LockInfo::None);
                }
                return State::Ok(Guard::new(Some(CowMutCell::Borrowed(Ref::map(
                    data,
                    |data| data.as_ref().unwrap(),
                )))));
            }
            Err(_) => return State::Err(LockInfo::Locked),
        }
    }

    pub fn get_mut<'a>(&'a self, id: usize) -> State<Guard<'a, T>> {
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
    use super::LockedArray;

    #[cfg(feature = "std")]
    #[test]
    fn test_set_and_get_multiple() {
        let locked_array = LockedArray::<Vec<i32>>::new();
        locked_array.set(0, vec![0, 0]);
        locked_array.set(1, vec![1]);
        locked_array.set(2, vec![2]);
        locked_array.set(3, vec![3]);

        let mut data0 = locked_array.get_mut(0).unwrap();
        assert_eq!(data0.as_slice(), [0, 0]);
        data0[0] = 1;
        assert_eq!(data0.as_slice(), [1, 0]);
        let mut data1 = locked_array.get_mut(1).unwrap();
        assert_eq!(data1.as_slice(), [1]);
        data1.push(2);
        assert_eq!(data1.as_slice(), [1, 2]);
    }
    
    #[cfg(feature = "std")]
    #[test]
    #[should_panic]
    fn test_set_same() {
        let locked_array = LockedArray::<Vec<i32>>::new();
        locked_array.set(1, vec![10]);
        locked_array.set(1, vec![10]);
    }
    
    #[cfg(feature = "std")]
    #[test]
    fn test_get_not_set() {
        let locked_array = LockedArray::<Vec<i32>>::new();
        {
            let _d = locked_array.get_mut(1);
            assert!(locked_array.get_mut(1).is_err());
        }
        let _ = locked_array.get_mut(1);
        assert!(locked_array.get_mut(1).is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_get_same_multiple() {
        let locked_array = LockedArray::<Vec<i32>>::new();
        locked_array.set(1, vec![10]);
        {
            let _d = locked_array.get_mut(1);
            assert!(locked_array.get_mut(1).is_err());
        }
        let _ = locked_array.get_mut(1);
        assert!(locked_array.get_mut(1).is_ok());
    }
}

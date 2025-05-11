use core::ops::Deref;
use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    hash::{BuildHasher, Hash, RandomState},
};

use crate::{LockInfo, State};

#[derive(Debug)]
pub struct LockedMap<K, V, S = RandomState> {
    data: RefCell<HashMap<K, Box<RefCell<V>>, S>>,
}

impl<K, T, S: Default> Default for LockedMap<K, T, S> {
    fn default() -> Self {
        Self {
            data: Default::default(),
        }
    }
}

impl<K, T, S: Default> LockedMap<K, T, S> {
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Default::default(),
        }
    }
}
impl<K, T, S: BuildHasher> LockedMap<K, T, S> {
    #[inline]
    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.borrow().is_empty()
    }

    #[inline]
    pub fn clear(&self) {
        self.data.borrow_mut().clear()
    }

    pub fn insert(&self, id: K, data: T)
    where
        K: Eq + Hash,
    {
        // let map = unsafe { &mut *self.data.get() };
        let mut map = self.data.borrow_mut();
        if map.contains_key(&id) {
            panic!()
        }
        map.insert(id, Box::new(RefCell::new(data)));
    }

    pub fn get(&self, id: &K) -> State<Ref<T>>
    where
        K: Eq + Hash,
    {
        let map = unsafe { &*self.data.as_ptr() };
        let entry = map.get(id).ok_or(LockInfo::None)?;
        (&**entry).try_borrow().map_err(|_| LockInfo::Locked)
    }

    pub fn get_mut(&self, id: &K) -> State<RefMut<T>>
    where
        K: Eq + Hash,
    {
        let map = unsafe { &*self.data.as_ptr() };
        let entry = map.get(id).ok_or(LockInfo::None)?;
        (&**entry).try_borrow_mut().map_err(|_| LockInfo::Locked)
    }
}

#[cfg(test)]
mod tests {
    use std::hash::BuildHasherDefault;

    use crate::{NoHasher, UniqueId};

    use super::LockedMap;

    #[test]
    fn test_locked_boxed() {
        let locked_map = LockedMap::<UniqueId, Vec<u32>, BuildHasherDefault<NoHasher>>::new();

        locked_map.insert(0, vec![1, 2, 3, 4]);

        let x = locked_map.get_mut(&0).unwrap();
        for i in 1..1000 {
            locked_map.insert(i, vec![i as u32, 2, 3, 4]);
        }
        println!("x: {x:?}");
        let z = locked_map.get_mut(&3).unwrap();
        println!("z: {z:?}");
    }
}

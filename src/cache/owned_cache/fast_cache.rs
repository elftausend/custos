use crate::{LockedMap, NoHasher, State, UniqueId};
use core::{
    any::Any,
    cell::{Ref, RefMut},
    hash::BuildHasherDefault,
};

use super::{Cache, DynAnyWrapper};

#[derive(Debug)]
pub struct FastCache<T = Box<dyn Any>> {
    pub nodes: LockedMap<UniqueId, T, BuildHasherDefault<NoHasher>>,
}

impl FastCache {
    #[inline]
    pub fn new() -> Self {
        FastCache {
            nodes: Default::default(),
        }
    }
}

impl<T> Default for FastCache<T> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
        }
    }
}

impl<T: DynAnyWrapper> Cache<T> for FastCache<T> {
    #[inline]
    fn get_mut(&self, id: UniqueId, _len: usize) -> State<RefMut<T>> {
        self.nodes.get_mut(&id)
    }

    #[inline]
    fn insert(&self, id: UniqueId, _len: usize, data: T) {
        self.nodes.insert(id, data);
    }

    #[inline]
    fn get(&self, id: UniqueId, _len: usize) -> State<Ref<T>> {
        self.nodes.get(&id)
    }
}

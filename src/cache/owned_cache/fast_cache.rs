use crate::{LockedMap, NoHasher, State, UniqueId};
use core::{
    any::Any,
    cell::{Ref, RefMut},
    hash::BuildHasherDefault,
};

use super::Cache;

#[derive(Default, Debug)]
pub struct FastCache {
    pub nodes: LockedMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
}

impl FastCache {
    #[inline]
    pub fn new() -> Self {
        FastCache::default()
    }
}

impl Cache<Box<dyn Any>> for FastCache {
    #[inline]
    fn get_mut(&self, id: UniqueId, _len: usize) -> State<RefMut<Box<dyn Any>>> {
        self.nodes.get_mut(&id)
    }

    #[inline]
    fn insert(&self, id: UniqueId, _len: usize, data: Box<dyn Any>) {
        self.nodes.insert(id, data);
    }

    #[inline]
    fn get(&self, id: UniqueId, _len: usize) -> State<Ref<Box<dyn Any>>> {
        self.nodes.get(&id)
    }
}

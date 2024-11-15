use core::{any::Any, cell::RefMut, hash::BuildHasherDefault};
use crate::{LockedMap, NoHasher, State, UniqueId};

use super::Cache2;

#[derive(Default)]
pub struct FastCache2 {
    pub nodes: LockedMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
}

impl Cache2<Box<dyn Any>> for FastCache2 {
    #[inline]
    fn get_mut(&self, id: UniqueId, _len: usize) -> State<RefMut<Box<dyn Any>>> {
        self.nodes.get_mut(&id)
    }

    fn insert(&self, id: UniqueId, _len: usize, data: Box<dyn Any>) {
        self.nodes.insert(id, data);
    }
}

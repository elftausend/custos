use crate::{LockedMap, State, UniqueId};
use core::{
    any::Any,
    cell::{Ref, RefMut},
};

use super::Cache;

#[derive(Default)]
pub struct LengthCache {
    pub nodes: LockedMap<(usize, UniqueId), Box<dyn Any>>,
}

impl Cache<Box<dyn Any>> for LengthCache {
    #[inline]
    fn get_mut(&self, id: UniqueId, _len: usize) -> State<RefMut<Box<dyn Any>>> {
        self.nodes.get_mut(&(_len, id))
    }

    #[inline]
    fn insert(&self, id: UniqueId, _len: usize, data: Box<dyn Any>) {
        self.nodes.insert((_len, id), data);
    }

    #[inline]
    fn get(&self, id: UniqueId, _len: usize) -> State<Ref<Box<dyn Any>>> {
        self.nodes.get(&(_len, id))
    }
}

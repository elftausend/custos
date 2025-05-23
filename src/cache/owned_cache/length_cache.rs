use crate::{LockedMap, State, UniqueId};
use core::{
    any::Any,
    cell::{Ref, RefMut},
};

use super::{Cache, DynAnyWrapper};

pub struct LengthCache<T = Box<dyn Any>> {
    pub nodes: LockedMap<(usize, UniqueId), T>,
}

impl<T> Default for LengthCache<T> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
        }
    }
}

impl<T: DynAnyWrapper> Cache for LengthCache<T> {
    type CachedValue = T;

    #[inline]
    fn get_mut(&self, id: UniqueId, _len: usize) -> State<RefMut<T>> {
        self.nodes.get_mut(&(_len, id))
    }

    #[inline]
    fn insert(&self, id: UniqueId, _len: usize, data: T) {
        self.nodes.insert((_len, id), data);
    }

    #[inline]
    fn get(&self, id: UniqueId, _len: usize) -> State<Ref<T>> {
        self.nodes.get(&(_len, id))
    }
}

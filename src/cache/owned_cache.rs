// mod fast_cache;
mod fast_cache2;
use core::cell::{Ref, RefMut};

// pub use fast_cache::*;
pub use fast_cache2::*;

// mod length_cache;
// pub use length_cache::*;

use super::{State, UniqueId};

pub trait Cache<T> {
    fn get_mut(&self, id: UniqueId, len: usize) -> State<RefMut<T>>;
    fn get(&self, id: UniqueId, len: usize) -> State<Ref<T>>;
    fn insert(&self, id: UniqueId, len: usize, data: T);
}

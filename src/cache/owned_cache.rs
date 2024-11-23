mod fast_cache;
mod length_cache;

use core::cell::{Ref, RefMut};

pub use fast_cache::*;
pub use length_cache::*;

use super::{State, UniqueId};

pub trait Cache<T> {
    fn get_mut(&self, id: UniqueId, len: usize) -> State<RefMut<T>>;
    fn get(&self, id: UniqueId, len: usize) -> State<Ref<T>>;
    fn insert(&self, id: UniqueId, len: usize, data: T);
}

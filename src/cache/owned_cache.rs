// mod fast_cache;
mod fast_cache2;
use core::cell::RefMut;

// pub use fast_cache::*;
pub use fast_cache2::*;

// mod length_cache;
// pub use length_cache::*;

use crate::{Alloc, ShallowCopy, Shape, UniqueId, Unit};

use super::State;

pub trait Cache2<T> {
    fn get_mut(&self, id: UniqueId, len: usize) -> State<RefMut<T>>;
    fn insert(&self, id: UniqueId, len: usize, data: T);
}

pub trait Cache {
    unsafe fn get<T, S, D>(
        &mut self,
        device: &D,
        id: UniqueId,
        len: usize,
        new_buf_callback: impl FnMut(UniqueId, &D::Base<T, S>),
    ) -> D::Base<T, S>
    where
        T: Unit,
        D: Alloc<T>,
        D::Base<T, S>: ShallowCopy,
        S: Shape;
}

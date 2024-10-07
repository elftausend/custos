mod fast_cache;
pub use fast_cache::*;

mod length_cache;
pub use length_cache::*;

use crate::{Alloc, Parents, ShallowCopy, Shape, UniqueId, Unit};

pub trait Cache {
    unsafe fn get<T, S, D, const N: usize>(
        &mut self,
        device: &D,
        id: UniqueId,
        len: usize,
        new_buf_callback: impl FnMut(UniqueId, &D::Base<T, S>),
        parents: impl Parents<N>,
    ) -> D::Base<T, S>
    where
        T: Unit,
        D: Alloc<T> + 'static,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape;
}

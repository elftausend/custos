use crate::Shape;

use super::{Alloc, Device};

pub trait Retrieve<D> {
    #[track_caller]
    fn retrieve<T, S: Shape>(&self, device: &D, len: usize) -> D::Data<T, S>
    where
        D: Alloc;
}

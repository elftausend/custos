use crate::{Id, Parents};

pub trait Argument<const N: usize> {
    fn maybe_parents(&self) -> Option<[Id; N]>;
}

impl<T: Parents<N>, const N: usize> Argument<N> for T {
    #[inline]
    fn maybe_parents(&self) -> Option<[Id; N]> {
        Some(self.ids())
    }
}

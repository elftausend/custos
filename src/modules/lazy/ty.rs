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

#[derive(Clone, Copy, Debug)]
pub enum Type {
    F32,
    I32,
}

pub trait Graphable {
    const TYPE: Type;
}

impl Graphable for f32 {
    const TYPE: Type = Type::F32;
}

impl Graphable for i32 {
    const TYPE: Type = Type::I32;
}

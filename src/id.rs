use core::ops::Deref;

pub trait HasId {
    fn id(&self) -> Id;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Id {
    /// The id of the buffer.
    pub id: u64,
    /// The amount of elements a corresponding [`Buffer`](crate::Buffer) has.
    pub len: usize,
}

impl Deref for Id {
    type Target = u64;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

impl HasId for Id {
    #[inline]
    fn id(&self) -> Id {
        *self
    }
}

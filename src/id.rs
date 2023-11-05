use core::ops::{Deref, DerefMut};

pub trait HasId {
    const HAS_NO_ID: bool = false;
    fn id(&self) -> Id;

    #[inline]
    fn maybe_id(&self) -> Option<Id> {
        Some(self.id())
    }
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

#[derive(Debug, Clone, Copy)]
pub struct NoId<T> {
    pub data: T,
}

impl<T> HasId for NoId<T> {
    const HAS_NO_ID: bool = true;
    #[inline]
    fn id(&self) -> Id {
        unimplemented!("This type is marked as a no-id.");
    }

    #[inline]
    fn maybe_id(&self) -> Option<Id> {
        None
    }
}

impl<T: 'static> From<T> for NoId<T> {
    #[inline]
    fn from(value: T) -> Self {
        NoId { data: value }
    }
}

pub trait AsNoId: Sized {
    fn no_id(self) -> NoId<Self>;
}

impl<T: Into<NoId<T>>> AsNoId for T {
    #[inline]
    fn no_id(self) -> NoId<Self> {
        self.into()
    }
}

impl<T> Deref for NoId<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for NoId<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

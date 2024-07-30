use core::ops::{Deref, DerefMut};

use crate::{Buffer, Device, Shape, Unit};

pub trait HasId {
    const HAS_NO_ID: bool = false;
    fn id(&self) -> Id;

    #[inline]
    fn maybe_id(&self) -> Option<Id> {
        Some(self.id())
    }

    // TODO: maybe move this to another trait -> `RequiresGrad`, probably needs to be added as trait bound for D::Data/Base
    #[inline]
    fn requires_grad(&self) -> bool {
        false
    }

    // TODO: maybe move this to another trait -> `RequiresGrad`, probably needs to be added as trait bound for D::Data/Base
    #[inline]
    fn set_requires_grad(&mut self, _requires_grad: bool) {}
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
    const HAS_NO_ID: bool = false;

    #[inline]
    fn id(&self) -> Id {
        *self
    }

    fn maybe_id(&self) -> Option<Id> {
        Some(self.id())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoId<T> {
    pub(crate) data: T,
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

pub trait AsNoId: Sized {
    fn no_id(self) -> NoId<Self>;
}

impl<T: Into<NoId<T>>> AsNoId for T {
    #[inline]
    fn no_id(self) -> NoId<Self> {
        self.into()
    }
}

pub trait BufAsNoId: Sized {
    fn buf_no_id(self) -> NoId<Self>;
}

impl<'a, T: Unit, D: Device, S: Shape> BufAsNoId for &Buffer<'a, T, D, S> {
    #[inline]
    fn buf_no_id(self) -> NoId<Self> {
        NoId { data: self }
    }
}

impl<'a, T: Unit, D: Device, S: Shape> BufAsNoId for &mut Buffer<'a, T, D, S> {
    #[inline]
    fn buf_no_id(self) -> NoId<Self> {
        NoId { data: self }
    }
}

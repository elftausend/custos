use core::{
    any::Any,
    ops::{Deref, DerefMut},
};

use crate::{Buffer, Device, DeviceError, Shape, UniqueId, UpdateArg};

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

impl<T> UpdateArg for NoId<T> {
    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn update_arg(
        _to_update: &mut Self,
        _id: Option<UniqueId>,
        _buffers: &mut std::collections::HashMap<
            crate::UniqueId,
            Box<dyn core::any::Any>,
            core::hash::BuildHasherDefault<crate::NoHasher>,
        >,
    ) -> crate::Result<()> {
        Ok(())
    }
}

impl<'a, T: 'static, D: Device + 'static, S: Shape + 'static> UpdateArg for &Buffer<'a, T, D, S> {
    #[cfg(not(feature = "no-std"))]
    fn update_arg(
        to_update: &mut Self,
        id: Option<UniqueId>,
        buffers: &mut std::collections::HashMap<
            crate::UniqueId,
            Box<dyn core::any::Any>,
            core::hash::BuildHasherDefault<crate::NoHasher>,
        >,
    ) -> crate::Result<()> {
        let buf = buffers
            .get(&id.unwrap())
            .ok_or(DeviceError::InvalidLazyBuf)?;
        *to_update = unsafe { &*(&**buf as *const dyn Any as *const Buffer<T, D, S>) };
        //    *self = buffers.get(&self.id()).unwrap().downcast_ref().unwrap();
        Ok(())
    }
}

impl<'a, T: 'static, D: Device + 'static, S: Shape + 'static> UpdateArg
    for &mut Buffer<'a, T, D, S>
{
    #[cfg(not(feature = "no-std"))]
    fn update_arg(
        to_update: &mut Self,
        id: Option<UniqueId>,
        buffers: &mut std::collections::HashMap<
            crate::UniqueId,
            Box<dyn core::any::Any>,
            core::hash::BuildHasherDefault<crate::NoHasher>,
        >,
    ) -> crate::Result<()> {
        let buf = buffers
            .get_mut(&id.unwrap())
            .ok_or(DeviceError::InvalidLazyBuf)?;
        *to_update = unsafe { &mut *(&mut **buf as *mut dyn Any as *mut Buffer<T, D, S>) };
        Ok(())
        //    *self = buffers.get(&self.id()).unwrap().downcast_ref().unwrap();
    }
}
/*impl<'a, T: 'static, D: Device + 'static, S: Shape + 'static> UpdateArg for &mut Buffer<'a, T, D, S> {
    fn update_arg(
        &mut self,
        buffers: &std::collections::HashMap<crate::UniqueId, Box<dyn core::any::Any>, core::hash::BuildHasherDefault<crate::NoHasher>>,
    ) {
        let buf = buffers.get(&self.id()).unwrap();
        *self = unsafe {&*(&**buf as *const dyn Any as *const Buffer<T, D, S>)};
    //    *self = buffers.get(&self.id()).unwrap().downcast_ref().unwrap();
    }
}*/
pub trait BufAsNoId: Sized {
    fn buf_no_id(self) -> NoId<Self>;
}

impl<'a, T, D: Device, S: Shape> BufAsNoId for &Buffer<'a, T, D, S> {
    #[inline]
    fn buf_no_id(self) -> NoId<Self> {
        NoId { data: self }
    }
}

impl<'a, T, D: Device, S: Shape> BufAsNoId for &mut Buffer<'a, T, D, S> {
    #[inline]
    fn buf_no_id(self) -> NoId<Self> {
        NoId { data: self }
    }
}

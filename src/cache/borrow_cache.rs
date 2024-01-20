use core::{
    any::Any,
    fmt::{Debug, Display},
    hash::BuildHasherDefault,
    mem::transmute,
};
use std::collections::HashMap;

use std::rc::Rc;

use super::NoHasher;
use crate::{flag::AllocFlag, Alloc, Buffer, Device, Id, Shape, UniqueId};

#[derive(Clone, Copy)]
pub enum CachingError {
    InvalidId,
    InvalidTypeInfo,
}

impl CachingError {
    pub fn as_str(&self) -> &'static str {
        match self {
            CachingError::InvalidId => "InvalidId: Invalid Buffer identifier.",
            CachingError::InvalidTypeInfo => "InvalidTypeInfo: Invalid type information provided for allocated Buffer. Does your specific operation use mixed types?",
        }
    }
}

impl Debug for CachingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Display for CachingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::error::Error for CachingError {}

pub(crate) type Buffers =
    HashMap<UniqueId, Box<dyn crate::ShallowCopyable>, BuildHasherDefault<NoHasher>>;

#[derive(Default)]
pub struct BorrowCache {
    pub(crate) cache: Buffers,
}

// TODO: make BorrowedCache unuseable without device (=> Static get methods with D: CacheReturn)
impl BorrowCache {
    pub fn add_or_get<'a, T, D, S>(&mut self, device: &'a D, id: Id) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        D: Alloc<T> + 'static,
        D::Data<T, S>: crate::ShallowCopy,
        S: Shape,
    {
        self.add_buf_once::<T, D, S>(device, id);

        let buf_any = self.cache.get(&id).unwrap();
        let buf_any = buf_any.as_any();
        buf_any.downcast_ref().unwrap()
    }

    pub fn add_or_get_mut<'a, T, D, S>(&mut self, device: &D, id: Id) -> &mut Buffer<'a, T, D, S>
    where
        T: 'static,
        D: Alloc<T> + 'static,
        D::Data<T, S>: crate::ShallowCopy,
        S: Shape,
    {
        self.add_buf_once::<T, D, S>(device, id);
        self.get_buf_mut(id).unwrap()
    }

    pub fn add_buf_once<T, D, S>(&mut self, device: &D, id: Id)
    where
        T: 'static,
        D: Alloc<T> + 'static,
        D::Data<T, S>: crate::ShallowCopy,
        S: Shape,
    {
        if self.cache.get(&id).is_some() {
            return;
        }

        self.add_buf::<T, D, S>(device, id)
    }

    pub fn add_buf<T, D, S>(&mut self, device: &D, id: Id)
    where
        T: 'static,
        D: Alloc<T> + 'static,
        D::Data<T, S>: crate::ShallowCopy,
        S: Shape,
    {
        // not using ::new, because this buf would get added to the cache of the device.
        // not anymore ?
        let buf = Buffer {
            data: device.base_to_data(device.alloc::<S>(id.len, AllocFlag::BorrowedCache)),
            device: Some(device),
        };

        let buf = unsafe { transmute::<_, Buffer<'static, T, D, S>>(buf) };
        self.cache.insert(*id, Box::new(buf));
    }

    #[inline]
    pub fn get_buf_with_dev<'a, 'b, T, D, S>(
        &'b self,
        id: Id,
        _dev: &'a D,
    ) -> Option<&'b Buffer<'a, T, D, S>>
    where
        T: 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        self.cache.get(&id)?.as_any().downcast_ref()
    }

    #[inline]
    pub fn get_buf<'a, T, D, S>(&self, id: Id) -> Result<&Buffer<'a, T, D, S>, CachingError>
    where
        T: 'static,
        D: Device + 'static,
        S: Shape,
    {
        self.cache
            .get(&id)
            .ok_or(CachingError::InvalidId)?
            .as_any()
            .downcast_ref()
            .ok_or(CachingError::InvalidTypeInfo)
    }

    #[inline]
    pub fn get_buf_mut<'a, T, D, S>(
        &mut self,
        id: Id,
    ) -> Result<&mut Buffer<'a, T, D, S>, CachingError>
    where
        T: 'static,
        D: Device + 'static,
        D::Data<T, S>: crate::ShallowCopy,
        S: Shape,
    {
        unsafe {
            transmute(
                self.cache
                    .get_mut(&id)
                    .ok_or(CachingError::InvalidId)?
                    .as_any_mut()
                    .downcast_mut::<Buffer<T, D, S>>()
                    .ok_or(CachingError::InvalidTypeInfo),
            )
        }
    }
}

#[cfg(test)]
mod tests {

    /*#[test]
    fn test_comp_error() {
        let device = CPU::<Base>::new();


        let a = {
            let mut cache = BorrowingCache::default();
            cache.add_or_get::<f32, CPU, ()>(&device, Id::new(10))
        };
    }*/

    /*#[test]
    fn test_get_borrowed() {
        let device = CPU::<Base>::default();
        let mut cache = BorrowCache::default();

        let (fid, sid, tid) = (
            Id::new_bumped(10),
            Id::new_bumped(10),
            Id::new_bumped(10),
        );

        cache.add_buf_once::<f32, _, ()>(&device, fid);
        cache.add_buf_once::<f32, _, ()>(&device, sid);
        cache.add_buf_once::<f32, _, ()>(&device, tid);

        let a = cache.get_buf::<f32, _, ()>(fid).unwrap();
        let b = cache.get_buf::<f32, _, ()>(fid).unwrap();

        assert_eq!(a.ptr, b.ptr);
    }*/
}

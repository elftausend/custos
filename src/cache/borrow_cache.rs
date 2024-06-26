use core::{
    any::Any,
    fmt::{Debug, Display},
    hash::BuildHasherDefault,
    marker::PhantomData,
    mem::transmute,
};
use std::collections::HashMap;

use super::NoHasher;
use crate::{flag::AllocFlag, Alloc, Buffer, Device, Id, Shape, UniqueId, Unit};

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

pub(crate) type AnyBuffers = HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>;

#[derive(Default)]
pub struct BorrowCacheLT<'dev> {
    pub(crate) cache: AnyBuffers,
    pd: PhantomData<&'dev dyn Any>,
}

impl<'dev> BorrowCacheLT<'dev> {
    pub fn add_buf_once<T, D, S>(&mut self, device: &'dev D, id: Id, new_buf: &mut bool)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        if self.cache.contains_key(&id) {
            return;
        }
        *new_buf = true;
        self.add_buf::<T, D, S>(device, id)
    }

    pub fn add_buf<T, D, S>(&mut self, device: &'dev D, id: Id)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        // not using ::new, because this buf would get added to the cache of the device.
        // not anymore ?
        let buf = Buffer {
            data: device.base_to_data(device.alloc::<S>(id.len, AllocFlag::BorrowedCache).unwrap()),
            device: Some(device),
        };

        let buf = unsafe { transmute::<_, Buffer<'static, T, D, S>>(buf) };
        self.cache.insert(*id, Box::new(buf));
    }

    #[inline]
    pub fn get_buf<T, D, S>(&self, id: Id) -> Result<&Buffer<'dev, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        self.cache
            .get(&id)
            .ok_or(CachingError::InvalidId)?
            .downcast_ref()
            .ok_or(CachingError::InvalidTypeInfo)
    }

    #[inline]
    pub fn get_buf_mut<'a, T, D, S>(
        &'a mut self,
        id: Id,
    ) -> Result<&'a mut Buffer<'dev, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        unsafe {
            transmute::<Result<&'a mut Buffer<'static, T, D, S>, CachingError>, _>(
                self.cache
                    .get_mut(&id)
                    .ok_or(CachingError::InvalidId)?
                    .downcast_mut::<Buffer<T, D, S>>()
                    .ok_or(CachingError::InvalidTypeInfo),
            )
        }
    }
}

#[derive(Default)]
pub struct BorrowCache {
    pub(crate) cache: AnyBuffers,
}

impl BorrowCache {
    // pub fn add_or_get<'a, T, D, S>(
    //     &mut self,
    //     device: &'a D,
    //     id: Id,
    //     new_buf: &mut bool,
    // ) -> &Buffer<'a, T, D, S>
    // where
    //     T: Unit + 'static,
    //     D: Alloc<T> + 'static,
    //     S: Shape,
    // {
    //     self.add_buf_once::<T, D, S>(device, id, new_buf);

    //     let buf_any = self.cache.get(&id).unwrap();
    //     buf_any.downcast_ref().unwrap()
    // }

    // pub fn add_or_get_mut<'a, T, D, S>(
    //     &mut self,
    //     device: &'a D,
    //     id: Id,
    //     new_buf: &mut bool,
    // ) -> &mut Buffer<'a, T, D, S>
    // where
    //     T: Unit + 'static,
    //     D: Alloc<T> + 'static,
    //     S: Shape,
    // {
    //     self.add_buf_once::<T, D, S>(device, id, new_buf);
    //     unsafe { self.get_buf_mut(id).unwrap() }
    // }

    pub fn add_buf_once<T, D, S>(&mut self, device: &D, id: Id, new_buf: &mut bool)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        if self.cache.contains_key(&id) {
            return;
        }
        *new_buf = true;
        self.add_buf::<T, D, S>(device, id)
    }

    pub fn add_buf<'a, T, D, S>(&'a mut self, device: &'a D, id: Id)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        // not using ::new, because this buf would get added to the cache of the device.
        // not anymore ?
        let buf = Buffer {
            data: device.base_to_data(device.alloc::<S>(id.len, AllocFlag::BorrowedCache).unwrap()),
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
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        self.cache.get(&id)?.downcast_ref()
    }

    #[inline]
    pub unsafe fn get_buf<'a, T, D, S>(&self, id: Id) -> Result<&Buffer<'a, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        self.cache
            .get(&id)
            .ok_or(CachingError::InvalidId)?
            .downcast_ref()
            .ok_or(CachingError::InvalidTypeInfo)
    }

    #[inline]
    pub unsafe fn get_buf_mut<'a, T, D, S>(
        &mut self,
        id: Id,
    ) -> Result<&mut Buffer<'a, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        unsafe {
            transmute(
                self.cache
                    .get_mut(&id)
                    .ok_or(CachingError::InvalidId)?
                    .downcast_mut::<Buffer<T, D, S>>()
                    .ok_or(CachingError::InvalidTypeInfo),
            )
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    #[cfg(feature = "cpu")]
    fn test_comp_error() {
        use crate::{Base, BorrowCache, Id, CPU};

        let mut cache = BorrowCache::default();

        let a = {
            let device = CPU::<Base>::new();
            cache.add_buf::<f32, _, ()>(&device, Id { id: 0, len: 10 });
            // drop(device);
            let mut new_buf = false;
            // cache.add_or_get::<f32, CPU, ()>(&device, Id { id: 0, len: 10}, &mut new_buf)
        };
        // cache.cache.get(&3);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_caching_of_borrowed_cached() {
        use crate::{Base, BorrowCache, Buffer, Id, CPU};

        let device = CPU::<Base>::default();
        let mut cache = BorrowCache::default();

        let (fid, sid, tid) = (
            Id { id: 0, len: 10 },
            Id { id: 1, len: 10 },
            Id { id: 2, len: 10 },
        );

        cache.add_buf_once::<f32, _, ()>(&device, fid, &mut false);
        cache.add_buf_once::<f32, _, ()>(&device, sid, &mut false);
        cache.add_buf_once::<f32, _, ()>(&device, tid, &mut false);

        let a: &Buffer = unsafe { cache.get_buf::<f32, _, ()>(fid).unwrap() };
        let b: &Buffer = unsafe { cache.get_buf::<f32, _, ()>(fid).unwrap() };

        assert_eq!(a.ptr, b.ptr);
    }
}

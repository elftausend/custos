use core::{
    any::Any,
    fmt::{Debug, Display},
    hash::BuildHasherDefault,
};
use std::collections::HashMap;

use super::{Downcast, NoHasher};
use crate::{Alloc, Buffer, Device, Id, Shape, UniqueId, Unit, flag::AllocFlag};

#[derive(Clone, Copy)]
pub enum CachingError {
    InvalidId,
    InvalidTypeInfo,
}

impl CachingError {
    pub fn as_str(&self) -> &'static str {
        match self {
            CachingError::InvalidId => "InvalidId: Invalid Buffer identifier.",
            CachingError::InvalidTypeInfo => {
                "InvalidTypeInfo: Invalid type information provided for allocated Buffer. Does your specific operation use mixed types?"
            }
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
pub struct BorrowCache {
    pub(crate) cache: AnyBuffers,
}

impl BorrowCache {
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

    pub fn add_buf<T, D, S>(&mut self, device: &D, id: Id)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        // not using ::new, because this buf would get added to the cache of the device.
        // not anymore ?
        let buf: Buffer<T, D, S> = Buffer {
            data: device
                .default_base_to_data_unbound(device.alloc::<S>(id.len, AllocFlag::None).unwrap()),
            device: None,
        };

        self.cache.insert(*id, Box::new(buf));
    }

    #[inline]
    pub fn get_buf_with_dev<'a, 'b, T, D, S>(
        &'b self,
        id: Id,
        _dev: &'a D,
    ) -> Option<&'b Buffer<'static, T, D, S>>
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        self.get_buf(_dev, id).ok()
    }

    #[inline]
    pub fn get_buf<'a, 'b, T, D, S>(
        &'a self,
        _device: &D,
        id: Id,
    ) -> Result<&'a Buffer<'static, T, D, S>, CachingError>
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
    pub fn get_buf_mut<'a, 'b, T, D, S>(
        &'a mut self,
        _device: &'b D,
        id: Id,
    ) -> Result<&'a mut Buffer<'static, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        self.cache
            .get_mut(&id)
            .ok_or(CachingError::InvalidId)?
            .downcast_mut()
            .ok_or(CachingError::InvalidTypeInfo)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    #[cfg(feature = "cpu")]
    fn test_comp_error() {
        use crate::{Base, BorrowCache, CPU, Cached, Id};

        let mut cache = BorrowCache::default();

        let _a = {
            let device = CPU::<Cached<Base>>::new();
            device.modules.cache.nodes.insert(0, Box::new(4));
            cache.add_buf::<f32, _, ()>(&device, Id { id: 0, len: 10 });
            cache
                .get_buf_mut::<f32, _, ()>(&device, Id { id: 0, len: 10 })
                .unwrap();
            // drop(device);
            let mut _new_buf = false;
            // cache.add_or_get::<f32, CPU, ()>(&device, Id { id: 0, len: 10}, &mut new_buf)
        };

        let device = CPU::<Cached<Base>>::new();
        let buf = cache
            .get_buf::<f32, _, ()>(&device, Id { id: 0, len: 10 })
            .unwrap();

        // let invalid_device = buf.device();
        // println!("invalid: {:?}", invalid_device.modules.cache.nodes);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_caching_of_borrowed_cached() {
        use crate::{Base, BorrowCache, Buffer, CPU, Id};

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

        let a: &Buffer = cache.get_buf::<f32, _, ()>(&device, fid).unwrap();
        let b: &Buffer = cache.get_buf::<f32, _, ()>(&device, fid).unwrap();

        assert_eq!(a.ptr, b.ptr);
    }
}

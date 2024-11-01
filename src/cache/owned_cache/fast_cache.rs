use core::{any::Any, hash::BuildHasherDefault};
use std::{collections::HashMap, sync::Arc};

use crate::{
    flag::AllocFlag, Alloc, Cache, Device, LockedMap, NoHasher, PtrType, ShallowCopy, Shape, UniqueId, Unit
};

pub struct FastCache {
    pub nodes: LockedMap<UniqueId, Arc<dyn Any>, BuildHasherDefault<NoHasher>>,
}

impl Default for FastCache {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Cache for FastCache {
    #[inline]
    unsafe fn get<T, S, D>(
        &mut self,
        device: &D,
        id: UniqueId,
        len: usize,
        new_buf_callback: impl FnMut(UniqueId, &D::Base<T, S>),
    ) -> D::Base<T, S>
    where
        T: Unit,
        D: Alloc<T>,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        self.get(device, id, len, new_buf_callback).unwrap()
    }
}

impl FastCache {
    #[inline]
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
        }
    }

    /// # Safety
    /// Lifetime of data must be at least as long as the lifetime of the cache (usually the device).
    #[inline]
    pub unsafe fn get<T, S, D>(
        &mut self,
        device: &D,
        id: UniqueId,
        len: usize,
        new_buf_callback: impl FnMut(UniqueId, &D::Base<T, S>),
    ) -> crate::Result<D::Base<T, S>>
    where
        T: Unit,
        D: Alloc<T>,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        let maybe_allocated = self.nodes.get(&id);
        match maybe_allocated {
            Ok(data) => {
                let data = unsafe {
                    data.downcast_ref::<D::Base<T, S>>()
                        .expect("Invalid request for data type!")
                        .shallow()
                };

                // TODO: not necessary, could add length to hashmap
                assert_eq!(data.size(), len, "Data size mismatch! Did you use e.g. if conditions in a (cursor) loop retrieving buffers with a different size?");
                Ok(data)
            }
            Err(e) => {
                match e {
                    crate::LockInfo::Locked => panic!("should return error"),
                    crate::LockInfo::None => {
                        unsafe { self.add_node(device, id, len, new_buf_callback) }
                    },
                }
            },
        }
    }

    unsafe fn add_node<T, S, D>(
        &self,
        device: &D,
        id: UniqueId,
        len: usize,
        mut callback: impl FnMut(UniqueId, &D::Base<T, S>),
    ) -> crate::Result<<D as Device>::Base<T, S>>
    where
        T: Unit,
        D: Alloc<T>,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        let data = device.alloc::<S>(len, AllocFlag::None)?;
        let shallow_data = unsafe { data.shallow() };

        callback(id, &shallow_data);
        self.nodes.insert(id, Arc::new(data));

        Ok(shallow_data)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    use super::FastCache;
    #[cfg(feature = "cpu")]
    use crate::{Base, Cached, CPU};

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cache_add_node() {
        let mut cache = FastCache::default();
        let device = CPU::<Cached<Base>>::new();

        assert_eq!(cache.nodes.len(), 0);

        let out = unsafe { cache.add_node::<f32, (), _>(&device, 0, 10, |_a, _b| ()) }.unwrap();

        assert_eq!(cache.nodes.len(), 1);
        assert_eq!(out.len, 10);

        let out1 = unsafe { cache.get::<f32, (), _>(&device, 1, 10, |_a, _b| ()) }.unwrap();
        assert_ne!(out.ptr, out1.ptr);
    }

    #[cfg(feature = "cpu")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_cache_get_reuse_based_on_location() {
        use crate::Cursor;

        let mut cache = FastCache::default();
        let device = CPU::<Base>::new();

        let mut prev = None;
        for _ in device.range(0..1000) {
            let out3 = unsafe { cache.get::<f32, (), _>(&device, 0, 10, |_a, _b| ()) }.unwrap();
            if prev.is_none() {
                prev = Some(out3.ptr);
            }
            assert_eq!(prev.unwrap(), out3.ptr);
            assert_eq!(cache.nodes.len(), 1);
            prev = Some(out3.ptr);
        }
        assert_eq!(cache.nodes.len(), 1);
    }

    #[cfg(feature = "cpu")]
    #[test]
    #[should_panic]
    fn test_cache_with_diffrent_length_return() {
        use crate::{Base, Buffer, Cursor, Retriever};

        let dev = CPU::<Cached<Base>>::new();

        for i in dev.range(10) {
            if i == 4 {
                // has assert inside, therefore, this line leads to a crash due tue mismatiching lengths
                let buf: Buffer<u8, _> = dev.retrieve(5, ()).unwrap();
                assert_eq!(buf.len, 5);
            } else {
                let _x: Buffer<u8, _> = dev.retrieve(3, ()).unwrap();
            }
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cache_with_cursor_range_overlap() {
        use crate::{Base, Buffer, Cursor, Retriever};

        let dev = CPU::<Cached<Base>>::new();

        for _i in dev.range(10) {
            let _x: Buffer<u8, _> = dev.retrieve(3, ()).unwrap();
        }

        assert_eq!(dev.cursor(), 1);

        for _i in dev.range(1..7) {
            let _x: Buffer<u8, _> = dev.retrieve(4, ()).unwrap();
        }
        assert_eq!(dev.cursor(), 2);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cache_with_cached_call() {
        use crate::{Base, Buffer, Cursor, Retriever};

        let dev = CPU::<Cached<Base>>::new();

        let mut _buf: Option<Buffer<f32, _>> = None;
        _buf = dev.retrieve(10, ()).ok();

        for _ in 0..10 {
            dev.cached(|| {
                let mut _buf: Option<Buffer<f32, _>> = None;
                _buf = dev.retrieve(10, ()).ok();
                _buf = dev.retrieve(10, ()).ok();
                _buf = dev.retrieve(10, ()).ok();
                let nodes = &dev.modules.cache.borrow().nodes;
                assert_eq!(nodes.len(), 4);
            });
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cache_with_cached_call2() {
        use crate::{Base, Buffer, Cursor, Retriever};

        let dev = CPU::<Cached<Base>>::new();

        let mut _buf: Option<Buffer<f32, _>> = None;
        _buf = dev.retrieve(10, ()).ok();

        for _ in 0..10 {
            dev.cached(|| {
                let _buf: Option<Buffer<f32, _>> = dev.retrieve(10, ()).ok();
            });
            dev.cached(|| {
                let _buf: Option<Buffer<f32, _>> = dev.retrieve(10, ()).ok();
            });
            dev.cached(|| {
                let _buf: Option<Buffer<f32, _>> = dev.retrieve(10, ()).ok();
            });
            let nodes = &dev.modules.cache.borrow().nodes;
            assert_eq!(nodes.len(), 2);
        }
    }
}

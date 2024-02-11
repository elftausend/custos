use core::{any::Any, hash::BuildHasherDefault};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    flag::AllocFlag, Alloc, BoxedShallowCopy, Cursor, Device, NoHasher, PtrType, ShallowCopy,
    Shape, UniqueId,
};

#[derive(Clone)]
pub struct Cache {
    pub nodes: HashMap<UniqueId, Rc<dyn BoxedShallowCopy>, BuildHasherDefault<NoHasher>>,
}

impl Default for Cache {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Cache {
    #[inline]
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
        }
    }

    /// # Safety
    /// Lifetime of data must be at least as long as the lifetime of the cache (usually the device).
    #[track_caller]
    #[inline]
    pub unsafe fn get<T, S, D>(
        &mut self,
        device: &D,
        len: usize,
        new_buf_callback: impl FnMut(usize, &D::Base<T, S>),
    ) -> D::Base<T, S>
    where
        D: Alloc<T> + Cursor + 'static,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        let maybe_allocated = self.nodes.get(&(device.cursor() as UniqueId));
        match maybe_allocated {
            Some(data) => {
                unsafe { device.bump_cursor() };
                let data = unsafe {
                    (**data).as_any()
                        .downcast_ref::<D::Base<T, S>>()
                        .expect("Invalid request for data type!")
                        .shallow()
                };

                // TODO: not necessary, could add length to hashmap
                assert_eq!(data.size(), len, "Data size mismatch! Did you use e.g. if conditions in a (cursor) loop retrieving buffers with a different size?");
                data
            }
            None => self.add_node(device, len, new_buf_callback),
        }
    }

    #[track_caller]
    fn add_node<T, S, D>(
        &mut self,
        device: &D,
        len: usize,
        mut callback: impl FnMut(usize, &D::Base<T, S>),
    ) -> <D as Device>::Base<T, S>
    where
        D: Alloc<T> + Cursor,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        let data = device.alloc::<S>(len, AllocFlag::None);
        let shallow_data = unsafe { data.shallow() };

        callback(device.cursor(), &shallow_data);
        self.nodes
            .insert(device.cursor() as UniqueId, Rc::new(data));
        unsafe { device.bump_cursor() };

        shallow_data
    }
}

#[cfg(test)]
mod tests {
    use super::Cache;
    use crate::{Base, Cached, CPU};

    #[cfg(feauture = "cpu")]
    #[test]
    fn test_cache_add_node() {
        let mut cache = Cache::default();
        let device = CPU::<Cached<Base>>::new();

        assert_eq!(cache.nodes.len(), 0);

        let out = cache.add_node::<f32, (), _>(&device, 10, || ());

        assert_eq!(cache.nodes.len(), 1);
        assert_eq!(out.len, 10);

        let out1 = unsafe { cache.get::<f32, (), _>(&device, 10, || ()) };
        assert_ne!(out.ptr, out1.ptr);
    }

    #[cfg(feauture = "cpu")]
    #[test]
    fn test_cache_get_at_different_locations() {
        let mut cache = Cache::default();
        let device = CPU::<crate::Cached<Base>>::new();

        assert_eq!(cache.nodes.len(), 0);

        let out1 = unsafe { cache.get::<f32, (), _>(&device, 10, || ()) };
        assert_eq!(cache.nodes.len(), 1);

        let out2 = unsafe { cache.get::<f32, (), _>(&device, 10, || ()) };

        assert_ne!(out1.ptr, out2.ptr);
        assert_eq!(cache.nodes.len(), 2);
    }

    #[cfg(feauture = "cpu")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_cache_get_reuse_based_on_location() {
        let mut cache = Cache::default();
        let device = CPU::<Base>::new();

        let mut prev = None;
        for _ in 0..1000 {
            let out3 = unsafe { cache.get::<f32, (), _>(&device, 10, || ()) };
            if prev.is_none() {
                prev = Some(out3.ptr);
            }
            assert_eq!(prev.unwrap(), out3.ptr);
            assert_eq!(cache.nodes.len(), 1);
            prev = Some(out3.ptr);
        }
        assert_eq!(cache.nodes.len(), 1);
    }
}

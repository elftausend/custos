use core::{hash::BuildHasherDefault, panic::Location};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    flag::AllocFlag, Alloc, Device, HashLocation, LocationHasher, ShallowCopy, Shape,
};

#[derive(Debug, Clone)]
pub struct Cache {
    pub nodes:
        HashMap<HashLocation<'static>, Rc<dyn core::any::Any>, BuildHasherDefault<LocationHasher>>,
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

    #[track_caller]
    #[inline]
    pub fn get<T, S, D>(&mut self, device: &D, len: usize, callback: fn()) -> D::Data<T, S>
    where
        D: Alloc<T> + 'static,
        D::Data<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        let maybe_allocated = self.nodes.get(&Location::caller().into());
        match maybe_allocated {
            Some(data) => unsafe { data.downcast_ref::<D::Data<T, S>>().unwrap().shallow() },
            None => self.add_node(device, len, callback),
        }
    }

    #[track_caller]
    fn add_node<T, S, D>(
        &mut self,
        device: &D,
        len: usize,
        callback: fn(),
    ) -> <D as Device>::Data<T, S>
    where
        D: Alloc<T>,
        D::Data<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        let data = device.alloc::<S>(len, AllocFlag::None);
        let shallow_data = unsafe { data.shallow() };

        self.nodes.insert(Location::caller().into(), Rc::new(data));
        callback();

        shallow_data
    }
}

#[cfg(test)]
mod tests {
    use super::Cache;
    use crate::{Base, CPU};

    #[test]
    fn test_cache_add_node() {
        let mut cache = Cache::default();
        let device = CPU::<Base>::new();

        assert_eq!(cache.nodes.len(), 0);

        let out = cache.add_node::<f32, (), _>(&device, 10, || ());

        assert_eq!(cache.nodes.len(), 1);
        assert_eq!(out.len, 10);

        let out1 = cache.get::<f32, (), _>(&device, 10, || ());
        assert_ne!(out.ptr, out1.ptr);
    }

    #[test]
    fn test_cache_get_at_different_locations() {
        let mut cache = Cache::default();
        let device = CPU::<Base>::new();

        assert_eq!(cache.nodes.len(), 0);

        let out1 = cache.get::<f32, (), _>(&device, 10, || ());
        assert_eq!(cache.nodes.len(), 1);

        let out2 = cache.get::<f32, (), _>(&device, 10, || ());

        assert_ne!(out1.ptr, out2.ptr);
        assert_eq!(cache.nodes.len(), 2);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_cache_get_reuse_based_on_location() {
        let mut cache = Cache::default();
        let device = CPU::<Base>::new();

        let mut prev = None;
        for _ in 0..1000 {
            let out3 = cache.get::<f32, (), _>(&device, 10, || ());
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

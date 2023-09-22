use core::{hash::BuildHasherDefault, panic::Location};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{flag::AllocFlag, Alloc, Device, HashLocation, LocationHasher, PtrConv, Shape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache<D: Device> {
    pub nodes:
        HashMap<HashLocation<'static>, Rc<D::Data<u8, ()>>, BuildHasherDefault<LocationHasher>>,
}

impl<D: Device> Default for Cache<D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<SD: Device> Cache<SD> {
    #[inline]
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
        }
    }

    #[track_caller]
    #[inline]
    pub fn get<T, S: Shape, D: Device + Alloc<T>>(
        &mut self,
        device: &D,
        alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> D::Data<T, S>
    where
        SD: PtrConv<D>,
        D: PtrConv<SD>,
    {
        let maybe_allocated = self.nodes.get(&Location::caller().into());
        match maybe_allocated {
            Some(data) => unsafe { SD::convert(data, AllocFlag::Wrapper) },
            None => self.add_node(device, alloc_fn),
        }
    }

    #[track_caller]
    pub fn add_node<T, S: Shape, D: Alloc<T>>(
        &mut self,
        device: &D,
        alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> D::Data<T, S>
    where
        D: PtrConv<SD>,
    {
        let data = alloc_fn(device, AllocFlag::Wrapper);
        // let data = device.alloc::<S>(len, AllocFlag::Wrapper);

        let untyped_ptr = unsafe { D::convert(&data, AllocFlag::None) };
        self.nodes
            .insert(Location::caller().into(), Rc::new(untyped_ptr));

        data
    }
}

#[cfg(test)]
mod tests {
    use super::Cache;
    use crate::{Alloc, Base, CPU};

    #[test]
    fn test_cache_add_node() {
        let mut cache = Cache::<CPU<Base>>::default();
        let device = CPU::<Base>::new();

        assert_eq!(cache.nodes.len(), 0);

        let out = cache.add_node::<f32, (), _>(&device, |device, alloc_flag| {
            device.alloc::<()>(10, alloc_flag)
        });

        assert_eq!(cache.nodes.len(), 1);
        assert_eq!(out.len, 10);

        let out1 = cache.get::<f32, (), _>(&device, |device, alloc_flag| {
            device.alloc::<()>(10, alloc_flag)
        });
        assert_ne!(out.ptr, out1.ptr);
    }

    #[test]
    fn test_cache_get_at_different_locations() {
        let mut cache = Cache::<CPU<Base>>::default();
        let device = CPU::<Base>::new();

        assert_eq!(cache.nodes.len(), 0);

        let out1 = cache.get::<f32, (), _>(&device, |device, alloc_flag| {
            device.alloc::<()>(10, alloc_flag)
        });
        assert_eq!(cache.nodes.len(), 1);

        let out2 = cache.get::<f32, (), _>(&device, |device, alloc_flag| {
            device.alloc::<()>(10, alloc_flag)
        });

        assert_ne!(out1.ptr, out2.ptr);
        assert_eq!(cache.nodes.len(), 2);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_cache_get_reuse_based_on_location() {
        let mut cache = Cache::<CPU<Base>>::default();
        let device = CPU::<Base>::new();

        let mut prev = None;
        for _ in 0..1000 {
            let out3 = cache.get::<f32, (), _>(&device, |device, alloc_flag| {
                device.alloc::<()>(10, alloc_flag)
            });
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

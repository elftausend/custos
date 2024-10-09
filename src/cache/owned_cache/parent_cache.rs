use core::{any::Any, hash::BuildHasherDefault};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    flag::AllocFlag, Alloc, Cache, Device, HasId, Id, NoHasher, Parents, PtrType, ShallowCopy,
    Shape, UniqueId, Unit,
};

#[derive(Clone)]
pub struct ParentCache {
    pub nodes: HashMap<(Vec<Id>, usize), Arc<dyn Any>>,
    pub locked: HashSet<UniqueId, BuildHasherDefault<NoHasher>>,
}

impl Default for ParentCache {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Cache for ParentCache {
    #[inline]
    unsafe fn get<T, S, D, const N: usize>(
        &mut self,
        device: &D,
        id: UniqueId,
        len: usize,
        new_buf_callback: impl FnMut(UniqueId, &D::Base<T, S>),
        parents: impl Parents<N>,
    ) -> Option<D::Base<T, S>>
    where
        T: Unit,
        D: Alloc<T> + 'static,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        self.get(device, id, len, new_buf_callback, parents)
    }

    #[inline]
    fn unlock_id(&mut self, id: UniqueId) {
        dbg!(id);
        self.locked.remove(&id);
    }
}

impl ParentCache {
    #[inline]
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
            locked: Default::default(),
        }
    }

    /// # Safety
    /// Lifetime of data must be at least as long as the lifetime of the cache (usually the device).
    #[inline]
    pub unsafe fn get<T, S, D, const N: usize>(
        &mut self,
        device: &D,
        id: UniqueId,
        len: usize,
        new_buf_callback: impl FnMut(UniqueId, &D::Base<T, S>),
        parents: impl Parents<N>,
    ) -> Option<D::Base<T, S>>
    where
        T: Unit,
        D: Alloc<T> + 'static,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        let maybe_allocated = self.nodes.get(&(parents.ids().to_vec(), len));
        match maybe_allocated {
            Some(data) => unsafe {
                let mut data = data
                    .downcast_ref::<D::Base<T, S>>()
                    .expect("Invalid request for data type!")
                    .shallow();

                data.set_flag(AllocFlag::Cached);

                if self.locked.contains(&data.id()) {
                    None
                } else {
                    Some(data)
                }
            },
            None => Some(unsafe { self.add_node(device, id, len, new_buf_callback, parents) }),
        }
    }

    unsafe fn add_node<T, S, D, const N: usize>(
        &mut self,
        device: &D,
        id: UniqueId,
        len: usize,
        mut callback: impl FnMut(UniqueId, &D::Base<T, S>),
        parents: impl Parents<N>,
    ) -> <D as Device>::Base<T, S>
    where
        T: Unit,
        D: Alloc<T>,
        D::Base<T, S>: ShallowCopy + 'static,
        S: Shape,
    {
        let data = device.alloc::<S>(len, AllocFlag::None).unwrap();
        let shallow_data = unsafe { data.shallow() };

        callback(id, &shallow_data);
        self.nodes
            .insert((parents.ids().to_vec(), len), Arc::new(data));
        self.locked.insert(*shallow_data.id());

        shallow_data
    }
}

/*
#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    use super::ParentCache;
    #[cfg(feature = "cpu")]
    use crate::{Base, Cached, CPU};

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cache_add_node() {
        use crate::{Device, HasId};

        let mut cache = ParentCache::default();
        let device = CPU::<Cached<Base, ParentCache>>::new();

        let parent = device.buffer([1, 2, 3,]);

        assert_eq!(cache.nodes.len(), 0);

        let out = unsafe { cache.add_node::<f32, (), _, 1>(&device, 0, 10, |_a, _b| (), &parent) };

        assert_eq!(cache.nodes.len(), 1);
        assert_eq!(out.len, 10);

        let out1 = unsafe { cache.get::<f32, (), _, 1>(&device, 1, 10, |_a, _b| (), out.id()) }.unwrap();
        assert_ne!(out.ptr, out1.ptr);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cache_get_reuse_based_on_parent() {
        use crate::Device;

        let mut cache = ParentCache::default();
        let device = CPU::<Base>::new();

        let parent = device.buffer([1, 2, 3,]);

        let mut prev = None;
        for _ in 0..1000 {
            let out3 = unsafe { cache.get::<f32, (), _, 1>(&device, 0, 10, |_a, _b| (), &parent) }.unwrap();
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
    fn test_cache_with_diffrent_length_return() {
        use crate::{Base, Buffer, Device, Retriever};

        let dev = CPU::<Cached<Base, ParentCache>>::new();
        let parent = dev.buffer([1, 2, 3,]);

        for i in 0..10 {
            if i == 4 {
                let buf: Buffer<u8, _> = dev.retrieve(5, &parent).unwrap();
                assert_eq!(buf.len, 5);
            } else {
                let _x: Buffer<u8, _> = dev.retrieve(3, &parent).unwrap();
            }
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_alloc_cycle() {
        use core::mem::ManuallyDrop;

        use crate::{Device, HasId, Retriever};

        let dev = CPU::<Cached<Base, ParentCache>>::new();
        let parent = dev.buffer([1, 2, 3,]);
        // let mut second = dev.buffer([1, 2, 3,]);
        for i in 0..10 {

            dbg!(i);
            let second1: crate::Buffer<i32, _> = dev.retrieve(5, &parent).unwrap();
            dbg!(&*second1.id());
            // let new: crate::Buffer<i32, _> = dev.retrieve(5, &second).unwrap();

            // second = second1;
            let mut second1 = ManuallyDrop::new(second1);
            unsafe { ManuallyDrop::drop(&mut second1) }
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cache_with_cursor_range_overlap() {
        use crate::{Base, Buffer, Cursor, Retriever};

        let dev = CPU::<Cached<Base, ParentCache>>::new();

        for _i in dev.range(10) {
            let _x: Buffer<u8, _> = dev.retrieve(3, ()).unwrap();
        }

        assert_eq!(dev.cursor(), 1);

        for _i in dev.range(1..7) {
            let _x: Buffer<u8, _> = dev.retrieve(4, ()).unwrap();
        }
        assert_eq!(dev.cursor(), 2);
    }
}
*/

//! Contains the [`Cache`]ing logic.

use core::{cell::RefMut, fmt::Debug, hash::BuildHasherDefault, ops::BitXor};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    flag::AllocFlag, shape::Shape, Alloc, Buffer, CacheAble, Device, GlobalCount, GraphReturn,
    Ident, PtrConv, PtrType,
};

/// This trait makes a device's [`Cache`] accessible and is implemented for all compute devices.
pub trait CacheReturn: GraphReturn<GlobalCount> {
    /// Returns a reference to a device's [`Cache`].
    fn cache(&self) -> core::cell::Ref<Cache<Self>>
    where
        Self: PtrConv;

    /// Returns a mutable reference to a device's [`Cache`].
    fn cache_mut(&self) -> RefMut<Cache<Self>>
    where
        Self: PtrConv;
}

const K: usize = 0x517cc1b727220a95;

/// An low-overhead [`Ident`] hasher using "FxHasher".
#[derive(Default)]
pub struct IdentHasher {
    hash: usize,
}

impl std::hash::Hasher for IdentHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("IdentHasher only hashes usize.")
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.hash = self.hash.rotate_left(5).bitxor(i).wrapping_mul(K);
    }
}

impl<D> CacheAble<D> for Cache<D>
where
    D: PtrConv + CacheReturn,
{
    #[cfg(not(feature = "realloc"))]
    #[inline]
    fn retrieve<T, S: Shape>(
        device: &D,
        len: usize,
        add_node: impl crate::AddGraph,
    ) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        device
            .cache_mut()
            .get(device, Ident::new(len), add_node, crate::bump_count)
    }

    #[cfg(feature = "realloc")]
    #[inline]
    fn retrieve<T, S: Shape>(
        device: &D,
        len: usize,
        _add_node: impl crate::AddGraph,
    ) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        Buffer::new(device, len)
    }

    #[inline]
    unsafe fn get_existing_buf<T, S: Shape>(device: &D, ident: Ident) -> Option<Buffer<T, D, S>> {
        let ptr = D::convert(device.cache().nodes.get(&ident)?, AllocFlag::Wrapper);

        Some(Buffer {
            ptr,
            device: Some(device),
            ident: Some(ident),
        })
    }

    #[inline]
    fn remove(device: &D, ident: Ident) {
        device.cache_mut().nodes.remove(&ident);
    }

    fn add_to_cache<T, S: Shape>(device: &D, ptr: &<D as Device>::Ptr<T, S>) -> Option<Ident> {
        device.graph_mut().add_leaf(ptr.size());
        let ident = Ident::new_bumped(ptr.size());
        let raw_ptr = unsafe { std::rc::Rc::new(D::convert(ptr, AllocFlag::Wrapper)) };
        device.cache_mut().nodes.insert(ident, raw_ptr);
        Some(ident)
    }
}

/// A cache for 'no-generic' raw pointers.
pub struct Cache<D: Device> {
    /// A map of all cached buffers using a custom hash function.
    pub nodes: HashMap<Ident, Rc<D::Ptr<u8, ()>>, BuildHasherDefault<IdentHasher>>,
}

impl<D: Device> Debug for Cache<D>
where
    D::Ptr<u8, ()>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache2")
            .field("cache", &self.nodes)
            .finish()
    }
}

impl<D: Device> Default for Cache<D>
where
    D::Ptr<u8, ()>: Default,
{
    #[inline]
    fn default() -> Self {
        Self {
            nodes: Default::default(),
        }
    }
}

impl<D: PtrConv + GraphReturn> Cache<D> {
    /// Adds a new cache entry to the cache.
    /// The next get call will return this entry if the [Ident] is correct.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::prelude::*;
    /// use custos::{Ident, bump_count};
    ///
    /// let device = CPU::new();
    /// let cache: Buffer = device
    ///     .cache_mut()
    ///     .add_node(&device, Ident { idx: 0, len: 7 }, (), bump_count);
    ///
    /// let ptr = device
    ///     .cache()
    ///     .nodes
    ///     .get(&Ident { idx: 0, len: 7 })
    ///     .unwrap()
    ///     .clone();
    ///
    /// assert_eq!(cache.host_ptr(), ptr.ptr as *mut f32);
    /// ```
    pub fn add_node<'a, T, S: Shape>(
        &mut self,
        device: &'a D,
        ident: Ident,
        _add_node: impl crate::AddGraph,
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S>,
    {
        let ptr = device.alloc(ident.len, AllocFlag::Wrapper);

        #[cfg(feature = "opt-cache")]
        let graph_node = device.graph_mut().add(ident.len, _add_node);

        #[cfg(not(feature = "opt-cache"))]
        let graph_node = crate::Node {
            idx: ident.idx,
            deps: [0; 2],
            len: ident.len,
        };

        let untyped_ptr = unsafe { D::convert(&ptr, AllocFlag::None) };
        self.nodes.insert(ident, Rc::new(untyped_ptr));

        Buffer {
            ptr,
            device: Some(device),
            ident: Some(Ident {
                idx: graph_node.idx,
                len: ident.len,
            }),
        }
    }

    /// Retrieves cached pointers and constructs a [`Buffer`] with the pointers and the given `len`gth.
    /// If a cached pointer doesn't exist, a new `Buffer` will be added to the cache and returned.
    ///
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::prelude::*;
    /// use custos::bump_count;
    ///
    /// let device = CPU::new();
    ///     
    /// let cache_entry: Buffer = device.cache_mut().get(&device, Ident::new(10), (), bump_count);
    /// let new_cache_entry: Buffer = device.cache_mut().get(&device, Ident::new(10), (), bump_count);
    ///
    /// assert_ne!(cache_entry.ptrs(), new_cache_entry.ptrs());
    ///
    /// unsafe { set_count(0) };
    ///
    /// let first_entry: Buffer = device.cache_mut().get(&device, Ident::new(10), (), bump_count);
    /// assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    /// ```
    pub fn get<'a, T, S: Shape>(
        &mut self,
        device: &'a D,
        ident: Ident,
        add_node: impl crate::AddGraph,
        callback: fn(),
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S>,
    {
        let may_allocated = self.nodes.get(&ident);

        callback();

        match may_allocated {
            Some(ptr) => {
                // callback();
                let typed_ptr = unsafe { D::convert(ptr, AllocFlag::Wrapper) };

                Buffer {
                    ptr: typed_ptr,
                    device: Some(device),
                    ident: Some(ident),
                }
            }
            None => self.add_node(device, ident, add_node),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::hash::Hasher;
    use std::collections::HashSet;

    //#[cfg(not(feature = "realloc"))]
    //use crate::set_count;
    //use crate::{bump_count, Buffer, CacheReturn, Ident, IdentHasher};

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_ident_hasher() {
        use crate::IdentHasher;

        let mut hashed_items = HashSet::new();
        let mut hasher = IdentHasher::default();

        for item in 0..2500000 {
            hasher.write_usize(item);
            hasher.write_usize(100000);
            let hashed_item = hasher.finish();
            assert!(!hashed_items.contains(&hashed_item));

            hashed_items.insert(hashed_item);
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_add_node() {
        use crate::{Buffer, CacheReturn, Ident};

        let device = crate::CPU::new();
        let cache: Buffer =
            device
                .cache_mut()
                .add_node(&device, Ident { idx: 0, len: 7 }, ());

        let ptr = device
            .cache()
            .nodes
            .get(&Ident { idx: 0, len: 7 })
            .unwrap()
            .clone();

        assert_eq!(cache.host_ptr(), ptr.ptr as *mut f32);
    }

    #[cfg(feature = "cpu")]
    #[cfg(not(feature = "realloc"))]
    #[test]
    fn test_get() {
        // for: cargo test -- --test-threads=1

        use crate::{set_count, Buffer, CacheReturn, Ident, bump_count};
        unsafe { set_count(0) };
        let device = crate::CPU::new();

        let cache_entry: Buffer = device.cache_mut().get(&device, Ident::new(10), (), bump_count);
        let new_cache_entry: Buffer = device.cache_mut().get(&device, Ident::new(10), (), bump_count);

        assert_ne!(cache_entry.ptrs(), new_cache_entry.ptrs());

        unsafe { set_count(0) };

        let first_entry: Buffer = device.cache_mut().get(&device, Ident::new(10), (), bump_count);
        assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    }
}

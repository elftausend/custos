use core::{cell::RefMut, hash::BuildHasherDefault, ops::BitXor};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    flag::AllocFlag, shape::Shape, Alloc, Buffer, CacheAble, Device, GraphReturn, Ident, PtrType,
};

/// This trait makes a device's [`Cache`] accessible and is implemented for all compute devices.
pub trait CacheReturn: GraphReturn {
    type CT;
    /// Returns a device specific [`Cache`].
    fn cache(&self) -> RefMut<Cache<Self>>
    where
        Self: RawConv;
}

pub trait RawConv: Device + CacheReturn {
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, flag: AllocFlag) -> Self::CT;
    fn destruct<T, S: Shape>(ct: &Self::CT) -> Self::Ptr<T, S>;
}

const K: usize = 0x517cc1b727220a95;

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

#[derive(Debug)]
pub struct Cache<D: RawConv> {
    pub nodes: HashMap<Ident, Rc<D::CT>, BuildHasherDefault<IdentHasher>>,
}

impl<D: RawConv> Default for Cache<D> {
    #[inline]
    fn default() -> Self {
        Self {
            nodes: Default::default(),
        }
    }
}

impl<D> CacheAble<D> for Cache<D>
where
    D: RawConv,
{
    #[cfg(not(feature = "realloc"))]
    #[inline]
    fn retrieve<T, S: Shape>(
        device: &D,
        len: usize, /*add_node: impl AddGraph*/
    ) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        device
            .cache()
            .get(device, Ident::new(len), crate::bump_count)
        //Cache::get(device, Ident::new(len), bump_count)
    }

    #[cfg(feature = "realloc")]
    #[inline]
    fn retrieve<T, S: Shape>(
        device: &D,
        len: usize, /*add_node: impl AddGraph*/
    ) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        Buffer::new(device, len)
    }

    #[inline]
    fn get_like<T, S: Shape>(device: &D, ident: Ident) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        device.cache().get(device, ident, || ())
    }

    #[inline]
    fn get_existing_buf<T, S: Shape>(device: &D, ident: Ident) -> Buffer<T, D, S> {
        let ptr = D::destruct::<T, S>(
            device
                .cache()
                .nodes
                .get(&ident)
                .expect("A matching Buffer does not exist."),
        );

        Buffer {
            ptr,
            device: Some(device),
            ident,
        }
    }

    #[inline]
    fn remove(device: &D, ident: Ident) {
        device.cache().nodes.remove(&ident);
    }

    fn add_to_cache<T, S: Shape>(device: &D, ptr: &<D as Device>::Ptr<T, S>) -> Ident {
        let ident = Ident::new_bumped(ptr.len());
        let raw_ptr = std::rc::Rc::new(D::construct(ptr, ptr.len(), AllocFlag::Wrapper));
        device.cache().nodes.insert(ident, raw_ptr);
        ident
    }
}

impl<D: RawConv> Cache<D> {
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
    ///     .cache()
    ///     .add_node(&device, Ident { idx: 0, len: 7 }, bump_count);
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
        callback: fn(),
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S> + RawConv,
    {
        let ptr = device.alloc(ident.len, AllocFlag::Cache);

        //#[cfg(feature = "opt-cache")]
        //let graph_node = device.graph().add(ident.len, _add_node);

        let raw_ptr = D::construct(&ptr, ident.len, AllocFlag::Cache);
        self.nodes.insert(ident, Rc::new(raw_ptr));

        callback();

        Buffer {
            ptr,
            device: Some(device),
            ident,
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
    /// let cache_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);
    /// let new_cache_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);
    ///
    /// assert_ne!(cache_entry.ptrs(), new_cache_entry.ptrs());
    ///
    /// unsafe { set_count(0) };
    ///
    /// let first_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);
    /// assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    /// ```
    pub fn get<'a, T, S: Shape>(
        &mut self,
        device: &'a D,
        ident: Ident,
        //add_node: impl AddGraph,
        callback: fn(),
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S> + RawConv,
    {
        let ptr_option = self.nodes.get(&ident);

        match ptr_option {
            Some(ptr) => {
                callback();

                let ptr = D::destruct::<T, S>(ptr);

                Buffer {
                    ptr,
                    device: Some(device),
                    ident,
                }
            }
            None => self.add_node(device, ident, callback),
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
        use crate::{bump_count, Buffer, CacheReturn, Ident};

        let device = crate::CPU::new();
        let cache: Buffer = device
            .cache()
            .add_node(&device, Ident { idx: 0, len: 7 }, bump_count);

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

        use crate::{bump_count, set_count, Buffer, CacheReturn, Ident};
        unsafe { set_count(0) };
        let device = crate::CPU::new();

        let cache_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);
        let new_cache_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);

        assert_ne!(cache_entry.ptrs(), new_cache_entry.ptrs());

        unsafe { set_count(0) };

        let first_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);
        assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    }
}

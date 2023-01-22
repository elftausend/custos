use core::{cell::RefMut, marker::PhantomData};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    bump_count, flag::AllocFlag, shape::Shape, Alloc, Buffer, CacheAble, Device, GraphReturn, Ident,
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
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize) -> Self::CT;
    fn destruct<T, S: Shape>(ct: &Self::CT, flag: AllocFlag) -> Self::Ptr<T, S>;
}

#[derive(Debug)]
pub struct Cache<D: RawConv> {
    pub nodes: HashMap<Ident, Rc<D::CT>>,
}

impl<D: RawConv> Default for Cache<D> {
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
    #[inline]
    fn retrieve<T, S: Shape>(
        device: &D,
        len: usize, /*add_node: impl AddGraph*/
    ) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        device.cache().get(device, Ident::new(len), bump_count)
        //Cache::get(device, Ident::new(len), bump_count)
    }

    #[inline]
    fn get_like<T, S: Shape>(device: &D, ident: Ident) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        device.cache().get(device, ident, || ())
    }
}

impl<D: RawConv> Cache<D> {
    /// Adds a new cache entry to the cache.
    /// The next get call will return this entry if the [Ident] is correct.
    /// # Example
    /// ```
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

        let raw_ptr = D::construct(&ptr, ident.len);
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
    /// ```
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
    /// set_count(0);
    ///
    /// let first_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);
    /// assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    /// ```
    #[cfg(not(feature = "realloc"))]
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

                let ptr = D::destruct::<T, S>(ptr, AllocFlag::Cache);

                Buffer {
                    ptr,
                    device: Some(device),
                    ident,
                }
            }
            None => self.add_node(device, ident, callback),
        }
    }

    /// If the 'realloc' feature is enabled, this functions always returns a new [`Buffer`] with the size of `len`gth.
    #[cfg(feature = "realloc")]
    #[inline]
    pub fn get<'a, T, S: Shape>(device: &'a D, len: usize, _: impl AddGraph) -> Buffer<T, D, S>
    where
        D: Alloc<'a, T, S>,
    {
        Buffer::new(device, len)
    }
}

#[cfg(feature = "cpu")]
#[cfg(test)]
mod tests {
    #[cfg(not(feature = "realloc"))]
    use crate::set_count;
    use crate::{bump_count, Buffer, CacheReturn, Ident};

    #[test]
    fn test_add_node() {
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

    #[cfg(not(feature = "realloc"))]
    #[test]
    fn test_get() {
        // for: cargo test -- --test-threads=1
        set_count(0);
        let device = crate::CPU::new();

        let cache_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);
        let new_cache_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);

        assert_ne!(cache_entry.ptrs(), new_cache_entry.ptrs());

        set_count(0);

        let first_entry: Buffer = device.cache().get(&device, Ident::new(10), bump_count);
        assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    }
}

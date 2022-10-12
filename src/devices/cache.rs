use crate::{
    bump_count, AddGraph, Alloc, BufFlag, Buffer, Device, GraphReturn, Ident, Node, PtrType,
};
use std::{cell::RefMut, collections::HashMap, ffi::c_void, rc::Rc};

pub trait CacheAble<D: Device, const N: usize = 0> {
    fn retrieve<T>(device: &D, len: usize, add_node: impl AddGraph) -> Buffer<T, D, N>
    where
        D: Alloc<T, N>;
}

/// This trait is implemented for every 'cacheable' pointer.
pub trait CacheType {
    /// Constructs a new device-specific cachable pointer.
    fn new<T>(ptr: (*mut T, *mut c_void, u64), len: usize, node: Node) -> Self;

    /// Destructs a device-specific pointer in its raw form.<br>
    /// See [CacheType::new].
    fn destruct<T>(&self) -> ((*mut T, *mut c_void, u64), Node);
}

/// This trait makes a device's [`Cache`] accessible and is implemented for all compute devices.
pub trait CacheReturn: GraphReturn {
    type CT: CacheType;
    /// Returns a device specific [`Cache`].
    fn cache(&self) -> RefMut<Cache<Self::CT>>;
}

/// Caches pointers that can be reconstructed into a [`Buffer`].
///
/// Example
///
/// ```
/// use custos::prelude::*;
///
/// let device = CPU::new();
/// let cached_buf = Cache::get::<f32, _, 0>(&device, 10, ());
///
/// assert_eq!(cached_buf.len, 10);
/// ```
#[derive(Debug)]
pub struct Cache<CT: CacheType> {
    pub nodes: HashMap<Ident, Rc<CT>>,
}

pub trait BindCT<CT> {}
impl<CT: CacheType> BindCT<CT> for Cache<CT> {}

impl<CT, D, const N: usize> CacheAble<D, N> for Cache<CT>
where
    D: Device + CacheReturn,
    CT: CacheType,
{
    fn retrieve<T>(device: &D, len: usize, add_node: impl AddGraph) -> Buffer<T, D, N>
    where
        D: Alloc<T, N>,
    {
        Cache::get(device, len, add_node)
    }
}

impl<CT: CacheType> Default for Cache<CT> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
        }
    }
}

impl<P: CacheType> Cache<P> {
    /// Adds a new cache entry to the cache.
    /// The next get call will return this entry if the [Ident] is correct.
    /// # Example
    /// ```
    /// use custos::prelude::*;
    /// use custos::Ident;
    ///
    /// let device = CPU::new();
    /// let cache: Buffer = device
    ///     .cache()
    ///     .add_node(&device, Ident { idx: 0, len: 7 }, ());
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
    pub fn add_node<'a, T, D, const N: usize>(
        &mut self,
        device: &'a D,
        node: Ident,
        _add_node: impl AddGraph,
    ) -> Buffer<'a, T, D, N>
    where
        D: Alloc<T, N> + GraphReturn,
    {
        let mut ptr = device.alloc(node.len);

        #[cfg(feature = "opt-cache")]
        let graph_node = device.graph().add(node.len, _add_node);

        #[cfg(not(feature = "opt-cache"))]
        let graph_node = Node::default();

        bump_count();

        self.nodes
            .insert(node, Rc::new(P::new(ptr.ptrs_mut(), node.len, graph_node)));

        Buffer {
            ptr,
            len: node.len,
            device: Some(device),
            flag: BufFlag::Cache,
            node: graph_node,
        }
    }

    /// Retrieves cached pointers and constructs a [`Buffer`] with the pointers and the given `len`gth.
    /// If a cached pointer doesn't exist, a new `Buffer` will be added to the cache and returned.
    ///
    /// # Example
    /// ```
    /// use custos::prelude::*;
    ///
    /// let device = CPU::new();
    ///     
    /// let cache_entry: Buffer = Cache::get(&device, 10, ());
    /// let new_cache_entry: Buffer = Cache::get(&device, 10, ());
    ///
    /// assert_ne!(cache_entry.ptrs(), new_cache_entry.ptrs());
    ///
    /// set_count(0);
    ///
    /// let first_entry: Buffer = Cache::get(&device, 10, ());
    /// assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    /// ```
    #[cfg(not(feature = "realloc"))]
    pub fn get<T, D, const N: usize>(
        device: &D,
        len: usize,
        add_node: impl AddGraph,
    ) -> Buffer<T, D, N>
    where
        // In order to know the specific pointer type
        // there is probably a better way to implement this
        Self: BindCT<D::CT>,
        D: Alloc<T, N> + CacheReturn,
    {
        let node = Ident::new(len);

        let mut cache = device.cache();
        let ptr_option = cache.nodes.get(&node);

        match ptr_option {
            Some(ptr) => {
                bump_count();
                let (ptr, node) = ptr.destruct::<T>();
                let ptr = <D as Device>::Ptr::<T, N>::from_ptrs(ptr);
                Buffer {
                    ptr,
                    len,
                    device: Some(device),
                    flag: BufFlag::Cache,
                    node,
                }
            }
            None => cache.add_node(device, node, add_node),
        }
    }

    /// If the 'realloc' feature is enabled, this functions always returns a new [`Buffer`] with the size of `len`gth.
    #[cfg(feature = "realloc")]
    pub fn get<T, D: Device, const N: usize>(
        device: &D,
        len: usize,
        _: impl AddGraph,
    ) -> Buffer<T, D, N>
    where
        // In order to know the specific pointer type
        // there is probably a better way to implement this
        Self: BindCT<D::CT>,
        D: Alloc<T, N> + CacheReturn,
    {
        Buffer::new(device, len)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "realloc"))]
    use crate::{set_count, Cache};
    use crate::{Buffer, CacheReturn, Ident, CPU};

    #[test]
    fn test_add_node() {
        let device = CPU::new();
        let cache: Buffer = device
            .cache()
            .add_node(&device, Ident { idx: 0, len: 7 }, ());

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
        let device = CPU::new();

        let cache_entry: Buffer = Cache::get(&device, 10, ());
        let new_cache_entry: Buffer = Cache::get(&device, 10, ());

        assert_ne!(cache_entry.ptrs(), new_cache_entry.ptrs());

        set_count(0);

        let first_entry: Buffer = Cache::get(&device, 10, ());
        assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    }
}

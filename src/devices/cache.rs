use crate::{Alloc, BufFlag, Buffer, Node, Graph};
use std::{cell::RefMut, collections::HashMap, ffi::c_void, marker::PhantomData};

/// This trait is implemented for every 'cacheable' pointer.
pub trait CacheType {
    /// Constructs a new device-specific cachable pointer.
    fn new<T>(ptr: (*mut T, *mut c_void, u64), len: usize) -> Self;

    /// Destructs a device-specific pointer in its raw form.<br>
    /// See [CacheType::new].
    fn destruct<T>(&self) -> (*mut T, *mut c_void, u64);
}

/// This trait makes a device's [`Cache`] accessible and is implemented for all compute devices.
pub trait CacheReturn<P: CacheType> {
    /// Returns a device specific [`Cache`].
    fn cache(&self) -> RefMut<Cache<P>>;
    fn graph(&self) -> RefMut<Graph>;
}

/// Caches pointers that can be reconstructed into a [`Buffer`].
#[derive(Debug)]
pub struct Cache<P: CacheType> {
    pub nodes: HashMap<Node, P>,
}

impl<P: CacheType> Default for Cache<P> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
        }
    }
}

impl<P: CacheType> Cache<P> {
    pub fn add_node<'a, T, D: Alloc<T> + CacheReturn<P>>(&mut self, device: &'a D, node: Node) -> Buffer<'a, T> {
        let ptr: (*mut T, *mut c_void, _) = device.alloc(node.len);

        let graph_node = device.graph().add_node(0, 0);

        self.nodes.insert(node, P::new(ptr, node.len));

        Buffer {
            ptr,
            len: node.len,
            device: Alloc::<T>::as_dev(device),
            flag: BufFlag::Cache(graph_node),
            p: PhantomData,
        }
    }

    /// Retrieves cached pointers and constructs a [`Buffer`] with them and `len`.
    #[cfg(not(feature = "realloc"))]
    pub fn get<T, D: Alloc<T> + CacheReturn<P>>(device: &D, len: usize) -> Buffer<T> {
        use crate::GNode;

        let node = Node::new(len);

        let mut cache = device.cache();
        let ptr_option = cache.nodes.get(&node);

        match ptr_option {
            Some(ptr) => Buffer {
                ptr: ptr.destruct(),
                len,
                device: Alloc::<T>::as_dev(device),
                flag: BufFlag::Cache(GNode { idx: 0, deps: [0, 0]}),
                p: PhantomData,
            },
            None => cache.add_node(device, node),
        }
    }

    #[cfg(feature = "realloc")]
    pub fn get<T, D: Alloc<T> + CacheReturn<P>>(device: &D, len: usize) -> Buffer<T> {
        Buffer::new(device, len)
    }
}

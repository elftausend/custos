use crate::{AddGraph, Alloc, BufFlag, Buffer, Node, GraphReturn, Ident, bump_count};
use std::{cell::RefMut, collections::HashMap, ffi::c_void, marker::PhantomData, rc::Rc};

/// This trait is implemented for every 'cacheable' pointer.
pub trait CacheType {
    /// Constructs a new device-specific cachable pointer.
    fn new<T>(ptr: (*mut T, *mut c_void, u64), len: usize, node: Node) -> Self;

    /// Destructs a device-specific pointer in its raw form.<br>
    /// See [CacheType::new].
    fn destruct<T>(&self) -> ((*mut T, *mut c_void, u64), Node);
}

/// This trait makes a device's [`Cache`] accessible and is implemented for all compute devices.
pub trait CacheReturn<P: CacheType>: GraphReturn {
    /// Returns a device specific [`Cache`].
    fn cache(&self) -> RefMut<Cache<P>>;
}

/// Caches pointers that can be reconstructed into a [`Buffer`].
#[derive(Debug)]
pub struct Cache<P: CacheType> {
    pub nodes: HashMap<Ident, Rc<P>>,
}

impl<P: CacheType> Default for Cache<P> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
        }
    }
}

impl<P: CacheType> Cache<P> {
    pub fn add_node<'a, T, D>(&mut self, device: &'a D, node: Ident, _add_node: impl AddGraph) -> Buffer<'a, T>
    where
        D: Alloc<T> + GraphReturn,
    {
        let ptr: (*mut T, *mut c_void, _) = device.alloc(node.len);

        #[cfg(feature="opt-cache")]
        let graph_node = device.graph().add(node.len, _add_node);

        #[cfg(not(feature="opt-cache"))]
        let graph_node = Node::default();

        bump_count();

        self.nodes
            .insert(node, Rc::new(P::new(ptr, node.len, graph_node)));

        Buffer {
            ptr,
            len: node.len,
            device: Alloc::<T>::as_dev(device),
            flag: BufFlag::Cache,
            node: graph_node,
            p: PhantomData,
        }
    }

    /// Retrieves cached pointers and constructs a [`Buffer`] with them and `len`.
    #[cfg(not(feature = "realloc"))]
    pub fn get<T, D>(device: &D, len: usize, add_node: impl AddGraph) -> Buffer<T>
    where
        D: Alloc<T> + CacheReturn<P>,
    {
        let node = Ident::new(len);
    
        let mut cache = device.cache();
        let ptr_option = cache.nodes.get(&node);

        match ptr_option {
            Some(ptr) => {
                bump_count();
                let (ptr, node) = ptr.destruct();
                Buffer {
                    ptr,
                    len,
                    device: Alloc::<T>::as_dev(device),
                    flag: BufFlag::Cache,
                    node,
                    p: PhantomData,
                }
            }
            None => cache.add_node(device, node, add_node),
        }
    }

    #[cfg(feature = "realloc")]
    pub fn get<T, D: Alloc<T> + CacheReturn<P>>(device: &D, len: usize, _: impl AddGraph) -> Buffer<T> {
        Buffer::new(device, len)
    }
}

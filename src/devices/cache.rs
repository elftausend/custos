use crate::{AddGraph, Alloc, BufFlag, Buffer, CacheTrace, GNode, GraphReturn, Node};
use std::{cell::RefMut, collections::HashMap, ffi::c_void, marker::PhantomData, rc::Rc};

/// This trait is implemented for every 'cacheable' pointer.
pub trait CacheType {
    /// Constructs a new device-specific cachable pointer.
    fn new<T>(ptr: (*mut T, *mut c_void, u64), len: usize, node: GNode) -> Self;

    /// Destructs a device-specific pointer in its raw form.<br>
    /// See [CacheType::new].
    fn destruct<T>(&self) -> ((*mut T, *mut c_void, u64), GNode);
}

/// This trait makes a device's [`Cache`] accessible and is implemented for all compute devices.
pub trait CacheReturn<P: CacheType>: GraphReturn {
    /// Returns a device specific [`Cache`].
    fn cache(&self) -> RefMut<Cache<P>>;
}

/// Caches pointers that can be reconstructed into a [`Buffer`].
#[derive(Debug)]
pub struct Cache<P: CacheType> {
    pub nodes: HashMap<Node, Rc<P>>,
    //pub cache_traces: Option<Vec<CacheTrace>>,
}

impl<P: CacheType> Default for Cache<P> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            //cache_traces: None,
        }
    }
}

impl<P: CacheType> Cache<P> {
    pub fn add_node<'a, T, D, A>(&mut self, device: &'a D, node: Node, add_node: A) -> Buffer<'a, T>
    where
        D: Alloc<T> + GraphReturn,
        A: AddGraph,
    {
        let ptr: (*mut T, *mut c_void, _) = device.alloc(node.len);

        let graph_node = device.graph().add(node.len, add_node);
        self.nodes.insert(node, Rc::new(P::new(ptr, node.len, graph_node)));

        Buffer {
            ptr,
            len: node.len,
            device: Alloc::<T>::as_dev(device),
            flag: BufFlag::Cache,
            node: graph_node,
            p: PhantomData,
        }
    }
/* 
    pub fn traced_buf<'a, T, D: Alloc<T>>(
        &self,
        device: &'a D,
        map_to: Node,
    ) -> Option<Buffer<'a, T>> {
       
        if let Some(cache_traces) = &self.cache_traces {
            for trace in cache_traces {
                if trace.use_cache_idx.contains(&map_to) {
                    let ptr = self
                        .nodes
                        .get(&Node {
                            idx: trace.cache_idx,
                            len: map_to.len,
                        })
                        .unwrap();
                    let (ptr, node) = ptr.destruct();

                    return Some(Buffer {
                        ptr,
                        len: node.len,
                        device: Alloc::<T>::as_dev(device),
                        flag: BufFlag::Cache,
                        node,
                        p: PhantomData,
                    });
                }
            }
        }
        None
    }
*/
    /// Retrieves cached pointers and constructs a [`Buffer`] with them and `len`.
    #[cfg(not(feature = "realloc"))]
    pub fn get<T, D, A>(device: &D, len: usize, add_node: A) -> Buffer<T>
    where
        D: Alloc<T> + CacheReturn<P>,
        A: AddGraph,
    {
        let node = Node::new(len);

        let mut cache = device.cache();

        //if let Some(cached) = cache.traced_buf(device, node) {
        //    return cached;
        //}

        let ptr_option = cache.nodes.get(&node);

        match ptr_option {
            Some(ptr) => {
                let (ptr, node) = ptr.destruct();
                Buffer {
                    ptr,
                    len: node.len,
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
    pub fn get<T, D: Alloc<T> + CacheReturn<P>>(device: &D, len: usize) -> Buffer<T> {
        Buffer::new(device, len)
    }
}

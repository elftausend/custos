use std::{ffi::c_void, collections::HashMap, marker::PhantomData, cell::RefMut};

use crate::{Node, Buffer, Alloc, BufFlag};

pub trait CacheType {
    fn new<T>(ptr: (*mut T, *mut c_void, u64), len: usize) -> Self;
    fn destruct<T>(&self) -> (*mut T, *mut c_void, u64);
}

pub trait CacheReturn<P: CacheType> {
    fn cache(&self) -> RefMut<Cache<P>>;
}

#[derive(Debug)]
pub struct Cache<P: CacheType> {
    pub nodes: HashMap<Node, P>
}

impl<P: CacheType> Default for Cache<P> {
    fn default() -> Self {
        Self { nodes: Default::default() }
    }
}

impl<P: CacheType> Cache<P> {
    pub fn add_node<'a, T, D: Alloc<T>>(&mut self, device: &'a D, node: Node) -> Buffer<'a, T> {
        let ptr: (*mut T, *mut c_void, _) = device.alloc(node.len);

        self.nodes.insert(node,
            P::new(ptr, 0)
        );
        Buffer {
            ptr,
            len: node.len,
            device: Alloc::<T>::as_dev(device),
            flag: BufFlag::Cache,
            p: PhantomData
        }
    }
    #[cfg(not(feature="realloc"))]
    pub fn get<T, D: Alloc<T>+CacheReturn<P>>(device: &D, len: usize) -> Buffer<T> {
        let node = Node::new(len);

        let mut cache = device.cache();
        let ptr_option = cache.nodes.get(&node);
        
        match ptr_option {
            Some(ptr) => {
                Buffer {
                    ptr: ptr.destruct(),
                    len,
                    device: Alloc::<T>::as_dev(device),
                    flag: BufFlag::Wrapper,
                    p: PhantomData,
                }                    
            }
            None => cache.add_node(device, node),
        }
    }
}
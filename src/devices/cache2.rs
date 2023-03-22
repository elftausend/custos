use core::{
    fmt::Debug,
    hash::BuildHasherDefault,
    mem::{align_of, size_of},
};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    cpu::CPUPtr, flag::AllocFlag, Alloc, Buffer, Cache2Return, CacheAble, CacheReturn, Device,
    Ident, IdentHasher, PtrType, Shape, CPU,
};

impl<D: Device + Cache2Return + PtrConv> CacheAble<D> for Cache2<D> {
    #[inline]
    fn retrieve<T, S: Shape>(
        device: &D,
        len: usize,
        add_node: impl crate::AddGraph,
    ) -> Buffer<T, D, S>
    where
        for<'a> D: Alloc<'a, T, S>,
    {
        device
            .cache_mut()
            .get(device, Ident::new(len), add_node, crate::bump_count)
    }

    #[inline]
    unsafe fn get_existing_buf<T, S: Shape>(device: &D, ident: Ident) -> Option<Buffer<T, D, S>> {
        let ptr = D::convert(device.cache().nodes.get(&ident)?, AllocFlag::Wrapper);

        Some(Buffer {
            ptr,
            device: Some(device),
            ident,
        })
    }

    #[inline]
    fn remove(device: &D, ident: Ident) {
        device.cache_mut().nodes.remove(&ident);
    }

    fn add_to_cache<T, S: Shape>(device: &D, ptr: &<D as Device>::Ptr<T, S>) -> Ident {
        device.graph_mut().add_leaf(ptr.size());
        let ident = Ident::new_bumped(ptr.size());
        let raw_ptr = unsafe { std::rc::Rc::new(D::convert(ptr, AllocFlag::Wrapper)) };
        device.cache_mut().nodes.insert(ident, raw_ptr);
        ident
    }
}

pub struct Cache2<D: Device> {
    pub nodes: HashMap<Ident, Rc<D::Ptr<u8, ()>>, BuildHasherDefault<IdentHasher>>,
}

impl<D: Device> Debug for Cache2<D>
where
    D::Ptr<u8, ()>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache2")
            .field("cache", &self.nodes)
            .finish()
    }
}

// Used to convert a device pointer to the a pointer of a different type.
pub trait PtrConv: Device + Cache2Return {
    /// Converts a pointer to a pointer with a different type.
    /// # Safety
    /// Prone to double frees. Make sure that the pointer is not freed twice.
    /// `custos` solves this by using fitting [`AllocFlag`]s.
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Ptr<T, IS>,
        flag: AllocFlag,
    ) -> Self::Ptr<Conv, OS>;
}

impl PtrConv for CPU {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Ptr<T, IS>,
        flag: AllocFlag,
    ) -> Self::Ptr<Conv, OS> {
        CPUPtr {
            ptr: ptr.ptr as *mut Conv,
            len: ptr.len,
            flag,
            align: Some(align_of::<T>()),
            size: Some(size_of::<T>()),
        }
    }
}

impl<D: Device> Default for Cache2<D>
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

impl<D: PtrConv> Cache2<D> {
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
    fn add_node<'a, T, S: Shape>(
        &mut self,
        device: &'a D,
        ident: Ident,
        _add_node: impl crate::AddGraph,
        callback: fn(),
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

        callback();

        Buffer {
            ptr,
            device: Some(device),
            ident: Ident {
                idx: graph_node.idx,
                len: ident.len,
            },
        }
    }

    fn get<'a, T, S: Shape>(
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

        match may_allocated {
            Some(ptr) => {
                callback();
                let typed_ptr = unsafe { D::convert(ptr, AllocFlag::Wrapper) };

                Buffer {
                    ptr: typed_ptr,
                    device: Some(device),
                    ident,
                }
            }
            None => self.add_node(device, ident, add_node, callback),
        }
    }
}

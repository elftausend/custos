use core::{
    cell::RefMut,
    marker::PhantomData,
    mem::{align_of, size_of, ManuallyDrop},
};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    bump_count,
    cpu::{alloc_initialized, CPUPtr, RawCpuBuf},
    flag::AllocFlag,
    opencl::RawCL,
    shape::Shape,
    AddGraph, Alloc, Buffer, CacheAble, Device, GraphReturn, Ident, Node, CacheAble2,
};

/// This trait makes a device's [`Cache`] accessible and is implemented for all compute devices.
pub trait CacheReturn: GraphReturn {
    type CT;
    /// Returns a device specific [`Cache`].
    fn cache(&self) -> RefMut<Cache<Self>>
    where
        Self: RawConv;
}

pub trait CacheReturn2: GraphReturn {
    /// Returns a device specific [`Cache`].
    fn cache(&self) -> RefMut<Cache2<Self>>
    where
        Self: BufType;
}

pub trait RawConv: Device + CacheReturn {
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, node: Node) -> Self::CT;
    fn destruct<T, S: Shape>(ct: &Self::CT, flag: AllocFlag) -> (Self::Ptr<T, S>, Node);
}

pub trait RawConv2: Device + CacheReturn {
    fn construct<T, S: Shape>(ptr: &Self::Ptr<u8, S>, len: usize, node: Node) -> Self::CT;
    fn destruct<T, S: Shape>(ct: &Self::CT, flag: AllocFlag) -> (Self::Ptr<T, S>, Node);
}

#[derive(Debug)]
pub struct Cache<D: RawConv> {
    pub nodes: HashMap<Ident, Rc<D::CT>>,
    _p: PhantomData<D>,
}

impl<D: RawConv> Default for Cache<D> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            _p: PhantomData,
        }
    }
}

pub trait BufType: Device {
    type Deallocator;

    unsafe fn ptr_to_raw<T, S: Shape>(ptr: &Self::Ptr<u8, S>) -> Self::Deallocator;
}

impl BufType for crate::CPU {
    type Deallocator = RawCpuBuf;

    unsafe fn ptr_to_raw<T, S: Shape>(ptr: &Self::Ptr<u8, S>) -> Self::Deallocator {
        RawCpuBuf {
            ptr: ptr.ptr,
            len: ptr.len,
            align: align_of::<T>(),
            size: size_of::<T>(),
            // FIXME: mind default node
            node: Node::default(),
        }
    }
}

#[cfg(feature = "opencl")]
impl BufType for crate::OpenCL {
    type Deallocator = RawCL;

    unsafe fn ptr_to_raw<T, S: Shape>(ptr: &Self::Ptr<u8, S>) -> Self::Deallocator {
        RawCL {
            ptr: ptr.ptr,
            host_ptr: ptr.host_ptr as *mut u8,
            len: ptr.len,
            // FIXME: mind default node
            node: Node::default(),
        }
    }
}

//#[derive(Debug)]
pub struct Cache2<'a, D: BufType = crate::CPU> {
    pub nodes: HashMap<Ident, (ManuallyDrop<Buffer<'a, u8, D, ()>>, D::Deallocator)>,
}

impl<'a, D: BufType> Cache2<'a, D> {
    fn add<T, S>(&mut self, device: &'a D, ident: Ident) -> &'a mut Buffer<'a, T, D, S>
    where
        S: Shape,
        D: for<'b> Alloc<'b, u8>,
    {
        let ptr = unsafe { device.alloc::<T>(ident.len, AllocFlag::Cache) };
        let raw = unsafe { D::ptr_to_raw::<T, ()>(&ptr) };

        let buf = ManuallyDrop::new(Buffer {
            ptr,
            device: Some(device),
            node: Node::default(),
        });

        self.nodes.insert(ident, (buf, raw));

        bump_count();

        unsafe {
            &mut *((&mut *self.nodes.get_mut(&ident).unwrap().0) as *mut Buffer<u8, D>).cast()
        }
    }

    fn get<T, S>(
        &mut self,
        device: &'a D,
        len: usize,
        _add_node: impl AddGraph,
    ) -> &'a mut Buffer<'a, T, D, S>
    where
        S: Shape,
        D: for<'b> Alloc<'b, u8>,
    {
        let ident = Ident::new(len);
        match self.nodes.get_mut(&ident) {
            Some(buf) => {
                bump_count();

                unsafe { &mut *(&mut *buf.0 as *mut Buffer<_, D>).cast() }
            }
            None => self.add(device, ident),
        }
    }
}

impl<D> CacheAble<D> for Cache<D>
where
    D: RawConv,
{
    #[inline]
    fn retrieve<T, S: Shape>(device: &D, len: usize, add_node: impl AddGraph) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        Cache::get(device, len, add_node)
    }
}

impl<D> CacheAble2<D> for Cache2<'_, D>
where
    D: BufType + CacheReturn2,
    D: for<'b> Alloc<'b, u8>
{
    #[inline]
    fn retrieve<'a, T, S: Shape>(device: &'a D, len: usize, add_node: impl AddGraph) -> &'a Buffer<'a, T, D, S>
    {
        device.cache().get::<T, S>(device, len, add_node)
    }
}

impl<D: RawConv> Cache<D> {
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
    pub fn add_node<'a, T, S: Shape>(
        &mut self,
        device: &'a D,
        node: Ident,
        _add_node: impl AddGraph,
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S> + RawConv,
    {
        let ptr = unsafe { device.alloc::<T>(node.len, AllocFlag::Cache) };

        #[cfg(feature = "opt-cache")]
        let graph_node = device.graph().add(node.len, _add_node);

        #[cfg(not(feature = "opt-cache"))]
        let graph_node = Node::default();

        let raw_ptr = D::construct(&ptr, node.len, graph_node);
        self.nodes.insert(node, Rc::new(raw_ptr));

        bump_count();

        Buffer {
            ptr,
            device: Some(device),
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
    pub fn get<'a, T, S: Shape>(
        device: &'a D,
        len: usize,
        add_node: impl AddGraph,
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S> + RawConv,
    {
        let node = Ident::new(len);

        let mut cache = device.cache();
        let ptr_option = cache.nodes.get(&node);

        match ptr_option {
            Some(ptr) => {
                bump_count();

                let (ptr, node) = D::destruct::<T, S>(ptr, AllocFlag::Cache);

                Buffer {
                    ptr,
                    device: Some(device),
                    node,
                }
            }
            None => cache.add_node(device, node, add_node),
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
    use std::collections::HashMap;

    #[cfg(not(feature = "realloc"))]
    use crate::{set_count, Cache};
    use crate::{Buffer, Cache2, CacheReturn, Device, Ident};

    pub struct Test<'a> {
        buf: Option<&'a Buffer<'a>>,
    }

    impl<'a> Test<'a> {
        pub fn forward(&mut self, buf: &'a Buffer) {
            self.buf = Some(buf);
        }
    }

    #[test]
    fn test_get2() {
        let device = crate::CPU::new();

        let mut cache: Cache2<crate::CPU> = Cache2 {
            nodes: HashMap::new(),
        };

        let mut test = Test { buf: None };

        for _x in 0..10 {
            let buf = cache.get::<f32, ()>(&device, 10, ());
            test.forward(buf);
        }
    }

    #[test]
    fn test_add_node() {
        let device = crate::CPU::new();
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
        let device = crate::CPU::new();

        let cache_entry: Buffer = Cache::get(&device, 10, ());
        let new_cache_entry: Buffer = Cache::get(&device, 10, ());

        assert_ne!(cache_entry.ptrs(), new_cache_entry.ptrs());

        set_count(0);

        let first_entry: Buffer = Cache::get(&device, 10, ());
        assert_eq!(cache_entry.ptrs(), first_entry.ptrs());
    }
}

use core::{cell::RefMut, marker::PhantomData, mem::{ManuallyDrop, align_of, size_of}};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    bump_count, flag::AllocFlag, shape::Shape, AddGraph, Alloc, Buffer, CacheAble, Device,
    GraphReturn, Ident, Node, cpu::{CPUPtr, alloc_initialized, RawCpuBuf},
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
    fn construct<T, S: Shape>(ptr: &Self::Ptr<T, S>, len: usize, node: Node) -> Self::CT;
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
    type Buf<'a>;
}

impl BufType for crate::CPU {
    type Buf<'a> = Buffer<'a, u8, crate::CPU>;
}

#[cfg(feature="opencl")]
impl BufType for crate::OpenCL {
    type Buf<'a> = Buffer<'a, u8, crate::OpenCL>;
}

#[derive(Debug)]
pub struct Cache2<'a, D: RawConv + BufType = crate::CPU> {
    pub nodes: HashMap<Ident, (ManuallyDrop<D::Buf<'a>>, D::CT)>
    //pub nodes: HashMap<Ident, &'a mut Buffer<u8>>
}

/*impl Drop for Cache2 {
    fn drop(&mut self) {
        todo!()
    }
}*/

#[test]
fn test_this() {
    for _ in 0..100000000000usize {
        // let val = ManuallyDrop::new(32i128);
        // assert_eq!(*val, 32);

        let val = ManuallyDrop::new(32i128);
        //assert_eq!(val[0], 32);
        assert_eq!(*val, 32);

        std::thread::sleep(std::time::Duration::from_micros(10))
    }
}

#[test]
fn test_leaking2() {
    let device = crate::CPU::new();

    let mut cache: Cache2<crate::CPU> = Cache2 {
        nodes: HashMap::new()
    };

    for _ in 0..1000 {
        cache.get::<f32>(&device, Ident::new(10));
    }

    drop(cache);
    drop(device);

    loop {

    }

}

#[test]
fn test_leaking() {
    let device = crate::CPU::new();

    let mut cache: Cache2<crate::CPU> = Cache2 {
        nodes: HashMap::new()
    };

    for _ in 0..1000 {
        cache.get::<f32>(&device, Ident::new(10));
    }
    //println!("{:?}", cache);

    /*for (_, (buf, _)) in &mut cache.nodes {
        unsafe {
            ManuallyDrop::drop(buf)
        }
    }*/

    drop(cache);
    drop(device);

    loop {

    }

}

impl<'c, /*D: BufType,*/> Cache2<'c, crate::CPU> {
    fn add<T, /*D: BufType*/>(&mut self, device: &'c crate::CPU, ident: Ident) -> &mut Buffer<T>
    where
        //D: for<'b> Alloc<'b, T>
    {
        //let ptr = device.alloc(ident.len, AllocFlag::Cache);
        //let ptr = CPUPtr::<u8>::new(ident.len * std::mem::size_of::<T>(), AllocFlag::Cache);
        
        let ptr = CPUPtr {
            ptr: alloc_initialized::<T>(ident.len),
            len: ident.len,
            flag: AllocFlag::Cache,
        };

        let raw = RawCpuBuf {
            ptr: ptr.ptr,
            len: ident.len,
            align: align_of::<T>(),
            size: size_of::<T>(),
            node: Node::default(),
        };
        
        let buf = ManuallyDrop::new(Buffer {
            ptr,
            device: Some(device),
            //device: None,
            node: Node::default(),
        });

        //let buf = &*buf;
        self.nodes.insert(ident, (buf, raw));
        
        bump_count();

        unsafe {
            &mut *((&mut *self.nodes.get_mut(&ident).unwrap().0) as *mut Buffer<u8>).cast()
        }
    }
    fn get<'a, T, /*D: BufType*/>(&'a mut self, device: &'c crate::CPU, ident: Ident) -> &'a mut Buffer<'a, T> {
        match self.nodes.get_mut(&ident) {
            Some(buf) => {
                bump_count();
                println!("test");
                unsafe {
                    &mut *(&mut *buf.0 as *mut Buffer<_>).cast()
                }
                /*unsafe {
                    //&mut *(*buf as *mut Buffer<T, D>)
                }*/
                //todo!()
                // unsafe {
                    // return &mut *(buf as *mut Buffer<u8> as *mut Buffer<T, D>)
                // }
            },
            None => {
                self.add(device, ident)
            },
        }
        //buf as *const Buffer<u8> as *const Buffer<T, D>;

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
        let ptr = device.alloc(node.len, AllocFlag::Cache);

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
    #[cfg(not(feature = "realloc"))]
    use crate::{set_count, Cache};
    use crate::{Buffer, CacheReturn, Ident};

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

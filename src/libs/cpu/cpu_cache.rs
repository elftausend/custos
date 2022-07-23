use crate::{Buffer, Node, CPU, Device, BufFlag};
use std::{cell::RefCell, collections::HashMap};

thread_local! {
    pub static CPU_CACHE: RefCell<CPUCache> = RefCell::new(CPUCache { nodes: HashMap::new() });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CpuPtr(pub *mut usize);

type RawInfo = (CpuPtr, usize);

#[cfg_attr(feature = "safe", doc = "```ignore")]
#[derive(Debug)]
/// stores output pointers
///
/// # Example
/// ```
/// use custos::{CPU, AsDev, Node, cpu::{CPU_CACHE, CPUCache}};
///
/// let device = CPU::new().select();
///
/// let out = CPUCache::get::<i16>(&device, 100*100);
///
/// let info = CPU_CACHE.with(|cache| {
///     let cache = cache.borrow();
///     let mut node = Node::new(100*100);
///     node.idx = 0; // to get the pointer of "out"
///     *cache.nodes.get(&node).unwrap()
/// });
/// assert!(info.0.0 == out.ptr.0 as *mut usize);
/// ```
pub struct CPUCache {
    pub nodes: HashMap<Node, RawInfo>,
}

impl CPUCache {
    pub fn add_node<T: Default + Copy>(&mut self, device: &CPU, node: Node) -> Buffer<T> {
        let out = Buffer {
            ptr: device.alloc(node.len),
            len: node.len,
            flag: BufFlag::Cache,
        };
        self.nodes
            .insert(node, (CpuPtr(out.ptr.0 as *mut usize), out.len));
        out
    }

    #[cfg(not(feature = "safe"))]
    pub fn get<T: Default + Copy>(device: &CPU, len: usize) -> Buffer<T> {
        //assert!(!device.cpu.borrow().ptrs.is_empty(), "no cpu allocations");

        use std::ptr::null_mut;
        let node = Node::new(len);
        CPU_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => Buffer {
                    ptr: (buf_info.0.0 as *mut T, null_mut(), 0),
                    len: buf_info.1,
                    flag: BufFlag::Cache,  
                },
                None => cache.add_node(device, node),
            }
        })
    }
    #[cfg(feature = "safe")]
    pub fn get<T: Default + Copy>(device: &CPU, len: usize) -> Buffer<T> {
        Buffer::new(device, len)
    }
}

use crate::{Device, Node, CPU, Valid, Buffer, BufFlag};
use std::{
    cell::RefCell,
    collections::HashMap,
    mem::align_of,
    rc::Rc, ptr::null_mut,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawCpu {
    pub ptr: *mut usize,
    pub len: usize,
    pub align: usize,
}

impl Drop for RawCpu {
    fn drop(&mut self) {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len * self.align);
            Box::from_raw(slice);
        }
    }
}

thread_local! {
    pub static CPU_CACHE: RefCell<CPUCache> = RefCell::new(CPUCache {nodes: HashMap::new()});
}

pub struct CPUCache {
    pub nodes: HashMap<Node, (RawCpu, Rc<Valid>)>,
}

impl CPUCache {
    pub fn add_node<T: Default + Copy>(&mut self, device: &CPU, node: Node) -> Buffer<T> {
        let ptr: (*mut T, _, _) = device.alloc(node.len);

        let valid = Rc::new(Valid);

        let buf = Buffer {
            ptr,
            len: node.len,
            flag: BufFlag::Cache(Rc::downgrade(&valid))
        };

        self.nodes.insert(node, (RawCpu {
            ptr: ptr.0 as *mut usize,
            len: node.len,
            align: align_of::<T>(),
        }, valid));

        buf
    }

    #[cfg(not(feature="realloc"))]
    pub fn get<T: Default + Copy>(device: &CPU, len: usize) -> Buffer<T> {
        //assert!(!device.cpu.borrow().ptrs.is_empty(), "no cpu allocations");

        let node = Node::new(len);
        CPU_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => {
                    Buffer {
                        ptr: (buf_info.0.ptr as *mut T, null_mut(), 0),
                        len: buf_info.0.len,
                        flag: BufFlag::Cache(Rc::downgrade(&buf_info.1))
                    }                    
                }
                None => cache.add_node(device, node),
            }
        })
    }

    #[cfg(feature="realloc")]
    pub fn get<T: Default + Copy>(device: &CPU, len: usize) -> Buffer<T> {
        Buffer::new(device, len)
    }
}

/*

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
/// let ptr = CPU_CACHE.with(|cache| {
///     let cache = cache.borrow();
///     let mut node = Node::new(100*100);
///     node.idx = 0; // to get the pointer of "out"
///     cache.nodes.get(&node).unwrap().ptr
/// });
/// assert!(ptr == out.ptr.0 as *mut usize);
/// ```
pub struct CPUCache {
    pub nodes: HashMap<Node, RawCpu>,
}

impl CPUCache {
    pub fn count() -> usize {
        CPU_CACHE.with(|cache| cache.borrow().nodes.len())
    }

    pub fn add_node<T: Default + Copy>(&mut self, device: &CPU, node: Node) -> Buffer<T> {
        let out = Buffer {
            ptr: device.alloc(node.len),
            len: node.len,
            flag: BufFlag::Cache,
        };
        self.nodes.insert(
            node,
            RawCpu {
                ptr: out.ptr.0 as *mut usize,
                len: out.len,
                align: align_of::<T>(),
            },
        );

        out
    }

    pub fn get<T: Default + Copy>(device: &CPU, len: usize) -> Buffer<T> {
        //assert!(!device.cpu.borrow().ptrs.is_empty(), "no cpu allocations");
        
        let node = Node::new(len);
        CPU_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => Buffer {
                    ptr: (buf_info.ptr as *mut T, null_mut(), 0),
                    len: buf_info.len,
                    flag: BufFlag::Cache,
                },
                None => cache.add_node(device, node),
            }
        })
    }
}
*/
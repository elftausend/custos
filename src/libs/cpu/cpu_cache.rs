use crate::{BufFlag, Buffer, Device, Node, CPU};
use std::{
    cell::Cell,
    cell::RefCell,
    collections::HashMap,
    ffi::c_void,
    mem::align_of,
    ptr::null_mut,
    rc::{Rc, Weak},
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
    pub static CACHE: RefCell<CPUCache> = RefCell::new(CPUCache {nodes: HashMap::new()});
}

pub struct CacheBuffer<T> {
    buf: Cell<Buffer<T>>,
    ptr: (Weak<RawCpu>, *mut c_void, u64),
}
impl<T> CacheBuffer<T> {
    pub fn new(ptr: (Weak<RawCpu>, *mut c_void, u64), len: usize) -> CacheBuffer<T> {
        CacheBuffer {
            buf: Cell::new(Buffer {
                ptr: (null_mut(), null_mut(), 0),
                len,
                flag: BufFlag::Cache,
            }),
            ptr,
        }
    }

    /*fn to_buf(self) -> Buffer<T> {
        let ptr = ( (*Rc::as_ref(&self.ptr.0.upgrade().expect("invalid ptr !!!"))).ptr as *mut T, self.ptr.1, self.ptr.2);
        Buffer { ptr, len: ..., flag: BufFlag::Cache }
    }*/

    fn as_buf(&self) -> &Buffer<T> {
        let ptr = (
            (*Rc::as_ref(&self.ptr.0.upgrade().expect("invalid ptr !!!"))).ptr as *mut T,
            self.ptr.1,
            self.ptr.2,
        );
        let buf = self.buf.as_ptr();
        unsafe {
            (*buf).ptr = ptr;
            &*buf
        }
    }
}

impl<T> std::ops::Deref for CacheBuffer<T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Self::Target {
        self.as_buf()
    }
}

impl<T> std::ops::DerefMut for CacheBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr = (
            (*Rc::as_ref(&self.ptr.0.upgrade().expect("invalid ptr !!!"))).ptr as *mut T,
            self.ptr.1,
            self.ptr.2,
        );
        let buf = self.buf.as_ptr();
        unsafe {
            (*buf).ptr = ptr;
            &mut *buf
        }
    }
}

pub struct CPUCache {
    pub nodes: HashMap<Node, Rc<RawCpu>>,
}

impl CPUCache {
    pub fn add_node<T: Default + Copy>(&mut self, device: &CPU, node: Node) -> CacheBuffer<T> {
        let ptr: (*mut T, _, _) = device.alloc(node.len);

        let raw_cpu = Rc::new(RawCpu {
            ptr: ptr.0 as *mut usize,
            len: node.len,
            align: align_of::<T>(),
        });
        let cb = CacheBuffer::new((Rc::downgrade(&raw_cpu), null_mut(), 0), node.len);

        self.nodes.insert(node, raw_cpu);

        cb
    }

    pub fn get<T: Default + Copy>(device: &CPU, len: usize) -> CacheBuffer<T> {
        //assert!(!device.cpu.borrow().ptrs.is_empty(), "no cpu allocations");

        let node = Node::new(len);
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => {
                    CacheBuffer::new((Rc::downgrade(buf_info), null_mut(), 0), node.len)
                }
                None => cache.add_node(device, node),
            }
        })
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
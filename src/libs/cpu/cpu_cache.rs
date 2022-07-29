use crate::{Device, Node, CPU, Buffer, BufFlag};
use std::{
    cell::RefCell,
    collections::HashMap,
    mem::{align_of, size_of},
    ptr::null_mut, alloc::Layout,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawCpu {
    pub ptr: *mut u8,
    len: usize,
    align: usize,
    size: usize,
    valid: *mut bool,
}

impl Drop for RawCpu {
    fn drop(&mut self) {
        unsafe {
            *self.valid = false;
            let layout = Layout::array::<u8>(self.len*self.size)
                .unwrap().align_to(self.align).unwrap();
            std::alloc::dealloc(self.ptr, layout);
        }
    }
}

thread_local! {
    pub static CPU_CACHE: RefCell<CPUCache> = RefCell::new(CPUCache {nodes: HashMap::new()});
}

pub struct CPUCache {
    pub nodes: HashMap<Node, RawCpu>,
}

impl CPUCache {
    pub fn add_node<T: Default + Copy>(&mut self, device: &CPU, node: Node) -> Buffer<T> {
        let ptr: (*mut T, _, _) = device.alloc(node.len);

        let valid = Box::leak(Box::new(true));

        let buf = Buffer {
            ptr,
            len: node.len,
            flag: BufFlag::Cache(valid)
        };

        self.nodes.insert(node, RawCpu {
            ptr: ptr.0 as *mut u8,
            len: node.len,
            align: align_of::<T>(),
            size: size_of::<T>(),
            valid
        });

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
                        ptr: (buf_info.ptr as *mut T, null_mut(), 0),
                        len: buf_info.len,
                        flag: BufFlag::Cache(buf_info.valid)
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
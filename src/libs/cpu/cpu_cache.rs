use std::{collections::HashMap, cell::RefCell};
use crate::{Buffer, Node};
use super::InternCPU;

thread_local! {
    pub static CPU_CACHE: RefCell<CPUCache> = RefCell::new(CPUCache { nodes: HashMap::new() });
}

#[derive(Debug, Clone, Copy)]
pub struct CpuPtr(pub *mut usize);

type RawInfo = (CpuPtr, usize);

#[derive(Debug)]
/// stores output pointers
/// 
/// # Example
/// ```
/// use custos::{Matrix, CPU, AsDev, cpu::CPU_CACHE, Node};
/// 
/// let device = CPU::new().select();
/// 
/// let a = Matrix::<i16>::new(&device, (100, 100));
/// let b = Matrix::<i16>::new(&device, (100, 100));
/// 
/// let out = a + b;
/// let info = CPU_CACHE.with(|cache| {
///     let cache = cache.borrow();
///     let mut node = Node::new(100*100);
///     node.idx = 0;
///     *cache.nodes.get(&node).unwrap()
/// });
/// assert!(info.0.0 == out.as_buf().ptr.0 as *mut usize);
/// ```
pub struct CPUCache {
    pub nodes: HashMap<Node, RawInfo>,
}

impl CPUCache {
    /*pub fn add_node<T: Default+Copy>(&mut self, device: InternCPU, node: Node) -> Matrix<T> {
        let out = Matrix::new(&device, node.out_dims);
        self.nodes.insert(node, ( CpuPtr(out.ptr() as *mut usize), out.dims() ));
        out
    }*/

    pub fn add_node<T: Default+Copy>(&mut self, device: InternCPU, node: Node) -> Buffer<T> {
        let out = Buffer::new(&device, node.len);
        self.nodes.insert(node, ( CpuPtr(out.ptr.0 as *mut usize), out.len ));
        out
    }
    
    #[cfg(not(feature="safe"))]
    pub fn get<T: Default+Copy>(device: InternCPU, len: usize) -> Buffer<T> {
        //assert!(!device.cpu.borrow().ptrs.is_empty(), "no cpu allocations");

        CPU_CACHE.with(|cache| {
            let node = Node::new(len);
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);
            
            match buf_info_option {
                Some(buf_info) => {
                    Buffer::from(( buf_info.0.0 as *mut T, buf_info.1 ))
                },
                None => cache.add_node(device, node)
            }
        })
    }
    #[cfg(feature="safe")]
    pub fn get<T: Default+Copy>(device: InternCPU, len: usize) -> Buffer<T> {
        Buffer::new(&device, len)
    }
}

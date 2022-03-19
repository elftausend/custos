use std::{collections::HashMap, sync::Mutex};

use crate::{libs::opencl::{GenericOCL, COUNT}, Matrix};

use super::CPU;

//pub static mut CPUCACHE_COUNT: usize = 0;


#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Node {
    idx: usize,
    out_dims: (usize, usize),
    thread_id: std::thread::ThreadId,
}

impl Node {
    pub fn new(out_dims: (usize, usize)) -> Node {
        let thread_id = std::thread::current().id();

        let mut guard = COUNT.lock().unwrap();

        let count = *guard.get(&thread_id).unwrap_or(&0);
        guard.insert(thread_id, count+1);
        
        Node {
            idx: count,
            out_dims,
            thread_id,
        }
    }
}

lazy_static::lazy_static! {
    pub static ref CPU_CACHE: Mutex<CPUCache> = Mutex::new(CPUCache { nodes: HashMap::new() });
}

#[derive(Debug, Clone, Copy)]
pub struct CpuPtr(*mut usize);

unsafe impl Sync for CpuPtr {}
unsafe impl Send for CpuPtr {}

type RawInfo = (CpuPtr, (usize, usize));

#[derive(Debug)]
pub struct CPUCache {
    nodes: HashMap<Node, RawInfo>,
}

impl CPUCache {
    pub fn add_node<T: GenericOCL>(&mut self, node: Node) -> Matrix<T> {
        let out = Matrix::new(CPU, node.out_dims);
        self.nodes.insert(node, ( CpuPtr(out.ptr() as *mut usize), out.dims() ));
        out

    }
    pub fn get<T: GenericOCL>(out_dims: (usize, usize)) -> Matrix<T> {
        let mut cache = CPU_CACHE.lock().unwrap();

        let node = Node::new(out_dims);
        let matrix_info_option = cache.nodes.get(&node);

        match matrix_info_option {
            Some(matrix_info) => Matrix::from((matrix_info.0.0 as *mut T, matrix_info.1)),
            None => cache.add_node(node)
        }
    }
}

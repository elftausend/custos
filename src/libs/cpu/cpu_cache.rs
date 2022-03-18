use std::collections::HashMap;

use crate::{libs::opencl::{CACHE_COUNT, GenericOCL}, Matrix};

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
        let node = Node {
            idx: unsafe {CACHE_COUNT},
            out_dims,
            thread_id: std::thread::current().id(),
        };
        unsafe {CACHE_COUNT+=1};
        node
    }
}

pub static mut CPU_CACHE: CPUCache = CPUCache { nodes: None };

type RawInfo = (*mut usize, (usize, usize));

#[derive(Debug)]
pub struct CPUCache {
    nodes: Option<HashMap<Node, RawInfo>>,
}

impl CPUCache {
    pub fn sync(&mut self) {
        if self.nodes.is_none() {
            self.nodes = Some(HashMap::new())
        }
    }
    pub fn add_node<T: GenericOCL>(&mut self, node: Node) -> Matrix<T> {
        let out = Matrix::new(CPU, node.out_dims);
        self.nodes.as_mut().unwrap().insert(node, (out.ptr() as *mut usize, out.dims()));
        out

    }
    pub fn get<T: GenericOCL>(out_dims: (usize, usize)) -> Matrix<T> {
        let node = Node::new(out_dims);
        let matrix_info_option = unsafe {
            CPU_CACHE.nodes.as_ref().unwrap().get(&node)
        };
        match matrix_info_option {
            Some(matrix_info) => Matrix::from((matrix_info.0 as *mut T, matrix_info.1)),
            None => unsafe {CPU_CACHE.add_node(node)}
        }
    }
}

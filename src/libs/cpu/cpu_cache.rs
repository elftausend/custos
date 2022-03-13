use std::collections::HashMap;

use crate::{Matrix, libs::opencl::{GenericOCL, CACHE_COUNT}};

use super::CPU;


//pub static mut CPUCACHE_COUNT: usize = 0;


#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Node {
    idx: usize,
    out_dims: (usize, usize),
}

impl Node {
    pub fn new(out_dims: (usize, usize)) -> Node {
        let node = Node {
            idx: unsafe {CACHE_COUNT},
            out_dims,
        };
        unsafe {CACHE_COUNT+=1};
        node
    }
}

pub static mut CPU_CACHE: CPUCache = CPUCache { nodes: None };

#[derive(Debug)]
pub struct CPUCache {
    nodes: Option<HashMap<Node, (*mut usize, (usize, usize))>>,
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
    pub fn get<T: GenericOCL>(node: Node) -> Matrix<T> {
        let matrix_info_option = unsafe {
            CPU_CACHE.nodes.as_ref().unwrap().get(&node)
        };
        match matrix_info_option {
            Some(matrix_info) => Matrix::from((matrix_info.0 as *mut T, matrix_info.1)),
            None => unsafe {CPU_CACHE.add_node(node)}
        }
    }
}

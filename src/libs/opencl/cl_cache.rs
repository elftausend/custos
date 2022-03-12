use std::{collections::HashMap, ffi::c_void};

use crate::{Buffer, matrix::Matrix};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Node {
    idx: usize,
    out_dims: (usize, usize),
    //lhs_dims: (usize, usize),
    //lhs_mem: Mem,
    //rhs_dims: (usize, usize),
    //rhs_mem: Mem,
}

static mut CLCACHE_COUNT: usize = 0;

impl Node {
    pub fn new<T>(out_dims: (usize, usize)) -> Node {
        let node = Node {
            idx: unsafe {CLCACHE_COUNT},
            out_dims,
            //lhs_dims: lhs.dims,
            //lhs_mem: lhs.data,
            //rhs_dims: rhs.dims,
            //rhs_mem: rhs.data
        };
        unsafe {CLCACHE_COUNT+=1};
        node
    }
}

pub static mut OCL_CACHE: OCLCache = OCLCache { nodes: None };

#[derive(Debug)]
pub struct OCLCache {
    nodes: Option<HashMap<Node, (*mut c_void, (usize, usize))>>,
}

impl OCLCache {
    pub fn sync(&mut self) {
        if self.nodes.is_none() {
            self.nodes = Some(HashMap::new())
        }
    }
    pub fn add_node<T: Default+Copy>(&mut self, node: Node) -> Matrix<T> {
        let out = Matrix::new(node.out_dims);
        self.nodes.as_mut().unwrap().insert(node, (out.ptr() as *mut c_void, out.dims()));
        out

    }
    pub fn get<T: Default+Copy>(node: Node) -> Matrix<T> {
        let matrix_info_option = unsafe {
            OCL_CACHE.nodes.as_ref().unwrap().get(&node)
        };
        match matrix_info_option {
            Some(matrix_info) => Matrix::from_ptr(matrix_info.0 as *mut T, matrix_info.1),
            None => unsafe {OCL_CACHE.add_node(node)}
        }
    }
}


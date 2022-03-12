use std::{collections::HashMap, ffi::c_void, any::TypeId};

use crate::{matrix::Matrix, number::Number};

use super::{api::{Kernel, create_program_with_source, build_program, create_kernels_in_program, set_kernel_arg}, CLDevice, GenericOCL};

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
    pub fn new(out_dims: (usize, usize)) -> Node {
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

pub static mut OCL_CACHE: OCLCache = OCLCache { output_nodes: None, arg_kernel_cache: None};

#[derive(Debug)]
pub struct OCLCache {
    output_nodes: Option<HashMap<Node, (*mut c_void, (usize, usize))>>,
    pub arg_kernel_cache: Option<HashMap<(Vec<*mut c_void>, Vec<TypeId>, Option<*mut c_void>, String), Kernel>>,
}

impl OCLCache {
    pub fn sync(&mut self) {
        if self.output_nodes.is_none() {
            self.output_nodes = Some(HashMap::new());
            self.arg_kernel_cache = Some(HashMap::new());
        }
    }
    pub fn add_node<T: GenericOCL>(&mut self, node: Node) -> Matrix<T> {
        let out = Matrix::new(node.out_dims);
        self.output_nodes.as_mut().unwrap().insert(node, (out.ptr() as *mut c_void, out.dims()));
        out

    }
    pub fn get<T: GenericOCL>(node: Node) -> Matrix<T> {
        let matrix_info_option = unsafe {
            OCL_CACHE.output_nodes.as_ref().unwrap().get(&node)
        };
        match matrix_info_option {
            Some(matrix_info) => Matrix::from_ptr(matrix_info.0 as *mut T, matrix_info.1),
            None => unsafe {OCL_CACHE.add_node(node)}
        }
    }

    pub fn arg_kernel_cache<'a, T: GenericOCL>(&mut self, device: CLDevice, tensors: &'a [(Matrix<T>, usize)], numbers: &'a [(T, usize)], output: Option<Matrix<T>>, src: String) -> Kernel {
        
        let mut mems = Vec::new();
        let type_ids = vec![TypeId::of::<T>(); numbers.len()];
        
        for tensor in tensors {
            mems.push(tensor.0.ptr() as *mut c_void)
        }

        let cache = self.arg_kernel_cache.as_mut().unwrap();
        let outputmem = output.map(|output| output.ptr() as *mut c_void);
        
        let kernel = cache.get(&(mems.clone(), type_ids.clone(), outputmem, src.clone()));
        match kernel {
            Some(kernel) => {
                kernel.clone()
            },
            None => {
                
                let program = create_program_with_source(&device.get_ctx(), &src).unwrap();
                build_program(&program, &[device.device], Some("-cl-std=CL1.2")).unwrap(); //-cl-single-precision-constant
                let kernel = &create_kernels_in_program(&program).unwrap()[0];
                for (number, idx) in numbers {
                    set_kernel_arg(kernel, *idx, number)
                }
                for (tensor, idx) in tensors {
                    set_kernel_arg(kernel, *idx, &(tensor.ptr() as *mut c_void));
                }
                if let Some(mem) = outputmem {
                    
                    set_kernel_arg(kernel, mems.len()+type_ids.len(), &mem);
                }

                cache.insert((mems, type_ids, outputmem, src), kernel.clone());
                kernel.clone()
            },
        }
        
    }
}


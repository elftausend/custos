use std::{any::TypeId, collections::HashMap, ffi::c_void};

use crate::matrix::Matrix;

use super::{api::{build_program, create_kernels_in_program, create_program_with_source, Kernel, set_kernel_arg}, CLDevice, GenericOCL};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Node {
    idx: usize,
    out_dims: (usize, usize),
    thread_id: std::thread::ThreadId,
}

// TODO Multithreading
pub static mut CACHE_COUNT: usize = 0;

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

pub static mut CL_CACHE: CLCache = CLCache { output_nodes: None, arg_kernel_cache: None};

type RawInfo = (*mut c_void, (usize, usize));
type KernelIdent = (std::thread::ThreadId, Vec<*mut c_void>, Vec<TypeId>, Option<*mut c_void>, String);

#[derive(Debug)]
pub struct CLCache {
    output_nodes: Option<HashMap<Node, RawInfo>>,
    pub arg_kernel_cache: Option<HashMap<KernelIdent, Kernel>>,
}

impl CLCache {
    pub fn sync(&mut self) {
        if self.output_nodes.is_none() {
            self.output_nodes = Some(HashMap::new());
            self.arg_kernel_cache = Some(HashMap::new());
        }
    }
    pub fn add_node<T: GenericOCL>(&mut self, device: CLDevice, node: Node) -> Matrix<T> {
        let out = Matrix::new(device, node.out_dims);
        self.output_nodes.as_mut().unwrap().insert(node, (out.ptr() as *mut c_void, out.dims()));
        out

    }
    pub fn get<T: GenericOCL>(device: CLDevice, node: Node) -> Matrix<T> {
        let matrix_info_option = unsafe {
            CL_CACHE.output_nodes.as_ref().unwrap().get(&node)
        };
        match matrix_info_option {
            Some(matrix_info) => Matrix::from((matrix_info.0 as *mut T, matrix_info.1)),
            None => unsafe {CL_CACHE.add_node(device, node)}
        }
    }

    pub fn arg_kernel_cache<T: GenericOCL>(&mut self, device: CLDevice, matrices: &[(Matrix<T>, usize)], numbers: &[(T, usize)], output: Option<Matrix<T>>, src: String) -> Kernel {
        let type_ids = vec![TypeId::of::<T>(); numbers.len()];
        
        let mems: Vec<*mut c_void> = matrices.iter()
            .map(|matrix| matrix.0.ptr() as *mut c_void)
            .collect();


        let cache = self.arg_kernel_cache.as_mut().unwrap();
        let outputmem = output.map(|output| output.ptr() as *mut c_void);
        
        let kernel = cache.get(&(std::thread::current().id(), mems.clone(), type_ids.clone(), outputmem, src.clone()));
        match kernel {
            Some(kernel) => kernel.clone(),
            None => {             
                let program = create_program_with_source(device.get_ctx(), &src).unwrap();
                build_program(&program, &[device.device], Some("-cl-std=CL1.2")).unwrap(); //-cl-single-precision-constant
                let kernel = &create_kernels_in_program(&program).unwrap()[0];
                
                for (number, idx) in numbers {
                    set_kernel_arg(kernel, *idx, number)
                }
                for (matrix, idx) in matrices {
                    set_kernel_arg(kernel, *idx, &(matrix.ptr() as *mut c_void));
                }
                if let Some(mem) = outputmem {
                    set_kernel_arg(kernel, mems.len()+type_ids.len(), &mem);
                }

                cache.insert((std::thread::current().id(), mems, type_ids, outputmem, src), kernel.clone());
                kernel.clone()
            },
        }
        
    }
}


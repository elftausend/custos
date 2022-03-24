use std::{any::TypeId, collections::HashMap, ffi::c_void, cell::RefCell};

use crate::matrix::Matrix;

use super::{api::{build_program, create_kernels_in_program, create_program_with_source, Kernel, set_kernel_arg}, CLDevice, GenericOCL};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Node {
    idx: usize,
    out_dims: (usize, usize),
    //pub thread_id: std::thread::ThreadId,
}

thread_local! {
    pub static COUNT: RefCell<usize> = RefCell::new(0);
}

pub fn set_count(count: usize) {
    COUNT.with(|c| *c.borrow_mut() = count);
}

pub fn get_count() -> usize {
    COUNT.with(|c| *c.borrow())
}

impl Node {
    pub fn new(out_dims: (usize, usize)) -> Node {
        COUNT.with(|count| {
            let node = Node {
                idx: *count.borrow(),
                out_dims,
                
            };
            *count.borrow_mut() += 1;
            node
        })
    }
}

/*
lazy_static::lazy_static! {
    #[derive(Debug)]
    pub static ref CL_CACHE: Mutex<CLCache> = Mutex::new(CLCache { output_nodes: HashMap::new(), arg_kernel_cache: HashMap::new() });
}*/

thread_local! {
    pub static CL_CACHE: RefCell<CLCache> = RefCell::new(CLCache { output_nodes: HashMap::new(), arg_kernel_cache: HashMap::new() })
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OclPtr(pub *mut c_void);

unsafe impl Send for OclPtr {}
unsafe impl Sync for OclPtr {}

type RawInfo = (OclPtr, (usize, usize));
type KernelIdent = (Vec<OclPtr>, Vec<TypeId>, Option<OclPtr>, String);

#[derive(Debug)]
pub struct CLCache {
    pub output_nodes: HashMap<Node, RawInfo>,
    pub arg_kernel_cache: HashMap<KernelIdent, Kernel>,
}

impl CLCache {
    pub fn add_node<T: GenericOCL>(&mut self, device: CLDevice, node: Node) -> Matrix<T> {
        let out = Matrix::new(device, node.out_dims);
        self.output_nodes.insert(node, ( OclPtr(out.ptr() as *mut c_void), out.dims() ));
        out

    }
    pub fn get<T: GenericOCL>(device: CLDevice, node: Node) -> Matrix<T> {
        CL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let matrix_info_option = cache.output_nodes.get(&node);
    
            match matrix_info_option {
                Some(matrix_info) => Matrix::from(( matrix_info.0.0 as *mut T, matrix_info.1 )),
                None => cache.add_node(device, node)
            }
        })
        /*
        let mut cache = CL_CACHE.lock().unwrap();

        let matrix_info_option = cache.output_nodes.get(&node);
    
        match matrix_info_option {
            Some(matrix_info) => Matrix::from(( matrix_info.0.0 as *mut T, matrix_info.1 )),
            None => cache.add_node(device, node)
        }
        */
    }

    pub fn arg_kernel_cache<T: GenericOCL>(&mut self, device: CLDevice, matrices: &[(Matrix<T>, usize)], numbers: &[(T, usize)], output: Option<Matrix<T>>, src: String) -> Kernel {
        let type_ids = vec![TypeId::of::<T>(); numbers.len()];
        
        let mems: Vec<OclPtr> = matrices.iter()
            .map(|matrix| OclPtr(matrix.0.ptr() as *mut c_void))
            .collect();

        let cache = &mut self.arg_kernel_cache;
        let outputmem = output.map(|output| OclPtr(output.ptr() as *mut c_void));
        
        let kernel = cache.get(&(mems.clone(), type_ids.clone(), outputmem, src.clone()));
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

                cache.insert((mems, type_ids, outputmem, src), kernel.clone());
                kernel.clone()
            },
        }
        
    }
}


use std::{any::TypeId, collections::HashMap, ffi::c_void, cell::RefCell};
use crate::{Error, GenericOCL};
use super::{api::{build_program, create_kernels_in_program, create_program_with_source, Kernel, set_kernel_arg}, cl_device::InternCLDevice};

#[cfg(feature="opencl")]
use crate::Buffer;

thread_local! {
    pub static CL_CACHE: RefCell<CLCache> = RefCell::new(CLCache { output_nodes: HashMap::new(), arg_kernel_cache: HashMap::new() })
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// A Node is used to identify a cached pointer.
pub struct Node {
    pub idx: usize,
    len: usize,
}

impl Node {
    pub fn new(len: usize) -> Node {
        crate::COUNT.with(|count| {
            let node = Node {
                idx: *count.borrow(),
                len,
            };
            *count.borrow_mut() += 1;
            node
        })
    }
}


#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OclPtr(pub *mut c_void);

unsafe impl Send for OclPtr {}
unsafe impl Sync for OclPtr {}

type RawInfo = (OclPtr, usize);
type KernelIdent = (Vec<OclPtr>, Vec<TypeId>, Option<OclPtr>, String);

#[derive(Debug)]
/// Stores kernels and outputs
pub struct CLCache {
    pub output_nodes: HashMap<Node, RawInfo>,
    pub arg_kernel_cache: HashMap<KernelIdent, Kernel>,
}

impl CLCache {
    pub fn add_node<T: GenericOCL>(&mut self, device: InternCLDevice, node: Node) -> Buffer<T> {
        let out = Buffer::new(&device, node.len);
        self.output_nodes.insert(node, ( OclPtr(out.ptr as *mut c_void), out.len ));
        out
    }

    #[cfg(not(feature="safe"))]
    pub fn get<T: GenericOCL>(device: InternCLDevice, node: Node) -> Buffer<T> {
        assert!(!device.cl.borrow().ptrs.is_empty(), "no OpenCL allocations");

        CL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.output_nodes.get(&node);
    
            match buf_info_option {
                Some(buf_info) => Buffer::from(( buf_info.0.0 as *mut T, buf_info.1 )),
                None => cache.add_node(device, node)
            }
        })
    }

    #[cfg(feature="safe")]
    pub fn get<T: GenericOCL>(device: InternCLDevice, node: Node) -> Buffer<T> {
        Buffer::new(&device, node.len)
    }

    pub fn arg_kernel_cache<T: GenericOCL>(&mut self, device: InternCLDevice, buffers: &[(&Buffer<T>, usize)], numbers: &[(T, usize)], output: Option<&Buffer<T>>, src: String) -> Result<Kernel, Error> {
        let type_ids = vec![TypeId::of::<T>(); numbers.len()];
        
        let mems: Vec<OclPtr> = buffers.iter()
            .map(|matrix| OclPtr(matrix.0.ptr as *mut c_void))
            .collect();

        let cache = &mut self.arg_kernel_cache;
        let outputmem = output.map(|output| OclPtr(output.ptr as *mut c_void));
        
        let kernel = cache.get(&(mems.clone(), type_ids.clone(), outputmem, src.clone()));
        match kernel { 
            Some(kernel) => Ok(kernel.clone()),
            None => {    
                let program = create_program_with_source(&device.ctx(), &src)?;
                build_program(&program, &[device.device()], Some("-cl-std=CL1.2"))?; //-cl-single-precision-constant
                let kernel = &create_kernels_in_program(&program)?[0];
                
                for (number, idx) in numbers {
                    set_kernel_arg(kernel, *idx, number)
                }

                for (buf, idx) in buffers {
                    set_kernel_arg(kernel, *idx, &(buf.ptr as *mut c_void));
                }

                if let Some(mem) = outputmem {
                    set_kernel_arg(kernel, mems.len()+type_ids.len(), &mem);
                }

                cache.insert((mems, type_ids, outputmem, src), kernel.clone());
                Ok(kernel.clone())
            },
        }
        
    }
}


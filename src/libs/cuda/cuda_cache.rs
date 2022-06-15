use std::{collections::HashMap, cell::RefCell};
use crate::{Node, InternCudaDevice, Buffer, Error};

use super::api::{FnHandle, nvrtc::create_program, load_module_data};

thread_local! {
    pub static CUDA_CACHE: RefCell<CudaCache> = RefCell::new(CudaCache { 
        kernels: HashMap::new(),
        nodes: HashMap::new(), 
    })
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct CudaPtr(pub u64);

unsafe impl Send for CudaPtr {}
unsafe impl Sync for CudaPtr {}

type RawInfo = (CudaPtr, usize);

pub struct CudaCache {
    pub kernels: HashMap<String, FnHandle>,
    pub nodes: HashMap<Node, RawInfo>,
}

impl CudaCache {
    pub fn add_node<T:>(&mut self, device: InternCudaDevice, node: Node) -> Buffer<T> {
        let out = Buffer::new(&device, node.len);
        self.nodes.insert(node, ( CudaPtr(out.ptr.2), out.len ));
        out
    }

    #[cfg(not(feature="safe"))]
    pub fn get<T: >(device: InternCudaDevice, len: usize) -> Buffer<T> {
        use std::ptr::null_mut;
        
        assert!(!device.cuda.borrow().ptrs.is_empty(), "no Cuda allocations");
        let node = Node::new(len);

        CUDA_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);
    
            match buf_info_option {
                Some(buf_info) => {
                    Buffer {
                        ptr: (null_mut(), null_mut(), buf_info.0.0),
                        len: buf_info.1
                    }
                }
                None => cache.add_node(device, node)
            }
        })
    }

    #[cfg(feature="safe")]
    pub fn get<T: GenericOCL>(device: InternCLDevice, len: usize) -> Buffer<T> {
        Buffer::new(&device, len)
    }

    pub fn kernel(&mut self, src: &str, fn_name: &str) -> Result<FnHandle, Error> {
        let kernel = self.kernels.get(src);

        if let Some(kernel) = kernel {
            return Ok(*kernel);
        }

        let x = create_program(&src, "")?;
        x.compile()?;
        let module = load_module_data(x.ptx()?)?;
        let function = module.function(fn_name)?;

        self.kernels.insert(src.into(), function);
        Ok(function)
    }
}


pub fn fn_cache(src: &str, fn_name: &str) -> crate::Result<FnHandle> {
    CUDA_CACHE.with(|cache| {
        cache.borrow_mut().kernel(src, fn_name)
    })
}   



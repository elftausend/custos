use super::api::{
    cufree, load_module_data,
    nvrtc::{create_program, nvrtcDestroyProgram},
    FnHandle,
};
use crate::{BufFlag, Buffer, CudaDevice, Error, Node, AsDev, Alloc};
use std::{cell::RefCell, collections::HashMap, ffi::CString, marker::PhantomData};

thread_local! {
    pub static CUDA_CACHE: RefCell<CudaCache> = RefCell::new(CudaCache {
        kernels: HashMap::new(),
        nodes: HashMap::new(),
    });
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct CudaPtr(pub u64);

unsafe impl Send for CudaPtr {}
unsafe impl Sync for CudaPtr {}

pub struct RawCUDA {
    pub ptr: u64,
}

impl Drop for RawCUDA {
    fn drop(&mut self) {
        unsafe { 
            cufree(self.ptr).unwrap() 
        }
    }
}

pub struct CudaCache {
    pub kernels: HashMap<String, FnHandle>,
    pub nodes: HashMap<Node, RawCUDA>,
}

impl CudaCache {
    pub fn add_node<'a, T>(&mut self, device: &'a CudaDevice, node: Node) -> Buffer<'a, T> {
        let ptr = device.alloc(node.len);
        
        self.nodes.insert(
            node,
            RawCUDA {
                ptr: ptr.2,
            },
        );
        Buffer {
            ptr,
            len: node.len,
            device: AsDev::dev(device),
            flag: BufFlag::Cache,
            p: PhantomData
        }
    }

    pub fn get<T>(device: &CudaDevice, len: usize) -> Buffer<T> {
        use std::ptr::null_mut;

        let node = Node::new(len);

        CUDA_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => Buffer {
                    ptr: (null_mut(), null_mut(), buf_info.ptr),
                    len,
                    device: AsDev::dev(device),
                    flag: BufFlag::Cache,
                    p: PhantomData
                },
                None => cache.add_node(device, node),
            }
        })
    }
    pub fn kernel(
        &mut self,
        device: &CudaDevice,
        src: &str,
        fn_name: &str,
    ) -> Result<FnHandle, Error> {
        let kernel = self.kernels.get(src);

        if let Some(kernel) = kernel {
            return Ok(*kernel);
        }

        let mut x = create_program(src, "")?;

        x.compile(Some(vec![CString::new("--use_fast_math").unwrap()]))?;

        let module = load_module_data(x.ptx()?)?;
        let function = module.function(fn_name)?;

        device.modules.borrow_mut().push(module);

        self.kernels.insert(src.into(), function);
        unsafe { nvrtcDestroyProgram(&mut x.0).to_result()? };
        Ok(function)
    }
}

pub fn fn_cache(device: &CudaDevice, src: &str, fn_name: &str) -> crate::Result<FnHandle> {
    CUDA_CACHE.with(|cache| cache.borrow_mut().kernel(device, src, fn_name))
}

use super::api::{
    load_module_data,
    nvrtc::{create_program, nvrtcDestroyProgram},
    FnHandle,
};
use crate::{Buffer, CudaDevice, Error, Node, Device, BufFlag};
use std::{cell::RefCell, collections::HashMap, ffi::CString};

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
    pub fn add_node<T>(&mut self, device: &CudaDevice, node: Node) -> Buffer<T> {
        let out = Buffer {
            ptr: device.alloc(node.len),
            len: node.len,
            flag: BufFlag::Cache
        };
        self.nodes.insert(node, (CudaPtr(out.ptr.2), out.len));
        out
    }

    #[cfg(not(feature = "safe"))]
    pub fn get<T>(device: &CudaDevice, len: usize) -> Buffer<T> {
        use std::ptr::null_mut;

        assert!(
            !device.inner.borrow().ptrs.is_empty(),
            "no Cuda allocations"
        );
        let node = Node::new(len);

        CUDA_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => Buffer {
                    ptr: (null_mut(), null_mut(), buf_info.0 .0),
                    len: buf_info.1,
                    flag: BufFlag::Cache
                },
                None => cache.add_node(device, node),
            }
        })
    }

    #[cfg(feature = "safe")]
    pub fn get<T>(device: &CudaDevice, len: usize) -> Buffer<T> {
        Buffer::new(device, len)
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

        let mut x = create_program(&src, "")?;

        x.compile(Some(vec![CString::new("--use_fast_math").unwrap()]))?;

        let module = load_module_data(x.ptx()?)?;
        let function = module.function(fn_name)?;

        device.inner.borrow_mut().modules.push(module);

        self.kernels.insert(src.into(), function);
        unsafe { nvrtcDestroyProgram(&mut x.0).to_result()? };
        Ok(function)
    }
}

pub fn fn_cache(device: &CudaDevice, src: &str, fn_name: &str) -> crate::Result<FnHandle> {
    CUDA_CACHE.with(|cache| cache.borrow_mut().kernel(device, src, fn_name))
}

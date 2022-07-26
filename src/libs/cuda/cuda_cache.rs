use super::api::{
    cufree, load_module_data,
    nvrtc::{create_program, nvrtcDestroyProgram},
    FnHandle,
};
use crate::{BufFlag, Buffer, CudaDevice, Device, Error, Node, Valid};
use std::{cell::RefCell, collections::HashMap, ffi::CString, rc::Rc};

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
        unsafe { cufree(self.ptr).unwrap() }
    }
}

pub struct CudaCache {
    pub kernels: HashMap<String, FnHandle>,
    pub nodes: HashMap<Node, (RawCUDA, Rc<Valid>)>,
}

impl CudaCache {
    pub fn count() -> usize {
        CUDA_CACHE.with(|cache| cache.borrow().nodes.len())
    }

    pub fn add_node<T>(&mut self, device: &CudaDevice, node: Node) -> Buffer<T> {
        let valid = Rc::new(Valid);

        let out = Buffer {
            ptr: device.alloc(node.len),
            len: node.len,
            flag: BufFlag::Cache(Rc::downgrade(&valid)),
        };
        self.nodes.insert(
            node,
            (RawCUDA {
                ptr: out.ptr.2,
            }, valid),
        );
        out
    }

    pub fn get<T>(device: &CudaDevice, len: usize) -> Buffer<T> {
        use std::ptr::null_mut;

        /*assert!(
            !device.inner.borrow().ptrs.is_empty(),
            "no Cuda allocations"
        );*/

        let node = Node::new(len);

        CUDA_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => Buffer {
                    ptr: (null_mut(), null_mut(), buf_info.0.ptr),
                    len,
                    flag: BufFlag::Cache(Rc::downgrade(&buf_info.1)),
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

        device.inner.borrow_mut().modules.push(module);

        self.kernels.insert(src.into(), function);
        unsafe { nvrtcDestroyProgram(&mut x.0).to_result()? };
        Ok(function)
    }
}

pub fn fn_cache(device: &CudaDevice, src: &str, fn_name: &str) -> crate::Result<FnHandle> {
    CUDA_CACHE.with(|cache| cache.borrow_mut().kernel(device, src, fn_name))
}

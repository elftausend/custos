use super::api::{
    build_program, create_kernels_in_program, create_program_with_source, release_mem_object, Kernel
};
use crate::{BufFlag, CLDevice, Device, Error, Node};
use std::{any::TypeId, cell::RefCell, collections::HashMap, ffi::c_void};

#[cfg(feature = "opencl")]
use crate::Buffer;

thread_local! {
    pub static CL_CACHE: RefCell<CLCache> = RefCell::new(CLCache {
        nodes: HashMap::new(),
        arg_kernel_cache: HashMap::new(),
        kernel_cache: HashMap::new()
    });
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OclPtr(pub *mut c_void);

unsafe impl Send for OclPtr {}
unsafe impl Sync for OclPtr {}

#[derive(Debug)]
pub struct RawCL {
    pub ptr: *mut c_void,
    pub host_ptr: *mut u8,
    pub valid: *mut bool
}

impl Drop for RawCL {
    fn drop(&mut self) {
        unsafe { 
            *self.valid = false;
            release_mem_object(self.ptr).unwrap() 
        };
    }
}

type KernelIdent = (Vec<OclPtr>, Vec<TypeId>, Option<OclPtr>, String);


#[derive(Debug)]
/// Stores kernels and outputs
pub struct CLCache {
    // TODO: Instead of a hashmap: vec?
    pub nodes: HashMap<Node, RawCL>,
    pub(crate) arg_kernel_cache: HashMap<KernelIdent, Kernel>,
    pub(crate) kernel_cache: HashMap<String, Kernel>,
}

impl CLCache {
    pub fn add_node<T>(&mut self, device: &CLDevice, node: Node) -> Buffer<T> {
        let ptr: (*mut T, *mut c_void, _) = device.alloc(node.len);

        let valid = Box::leak(Box::new(true));

        let out = Buffer {
            ptr,
            len: node.len,
            flag: BufFlag::Cache(valid as *const bool),
        };

        self.nodes.insert(node,
            RawCL {
                ptr: ptr.1,
                host_ptr: ptr.0 as *mut u8,
                valid
            }
        );
        out
    }

    pub fn get<T>(device: &CLDevice, len: usize) -> Buffer<T> {
        /*assert!(
            !device.inner.borrow().ptrs.is_empty(),
            "no OpenCL allocations"
        );*/
        let node = Node::new(len);

        CL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => Buffer {
                    ptr: (buf_info.host_ptr as *mut T, buf_info.ptr, 0),
                    len,
                    flag: BufFlag::Cache(buf_info.valid)
                },
                None => cache.add_node(device, node),
            }
        })
    }

    pub fn arg_kernel_cache(&mut self, device: &CLDevice, src: String) -> Result<Kernel, Error> {
        let kernel = self.kernel_cache.get(&src);

        if let Some(kernel) = kernel {
            return Ok(*kernel);
        }

        let program = create_program_with_source(&device.ctx(), &src)?;
        build_program(&program, &[device.device()], Some("-cl-std=CL1.2"))?; //-cl-single-precision-constant
        let kernel = create_kernels_in_program(&program)?[0];

        self.kernel_cache.insert(src, kernel);
        Ok(kernel)
    }
}

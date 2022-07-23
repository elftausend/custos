use super::api::{
    build_program, create_kernels_in_program, create_program_with_source, release_mem_object,
    set_kernel_arg, Kernel,
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
    pub len: usize,
}

impl Drop for RawCL {
    fn drop(&mut self) {
        unsafe { release_mem_object(self.ptr).unwrap() };
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
    pub fn count() -> usize {
        CL_CACHE.with(|cache| cache.borrow().nodes.len())
    }

    pub fn add_node<T>(&mut self, device: &CLDevice, node: Node) -> Buffer<T> {
        let out = Buffer {
            ptr: device.alloc(node.len),
            len: node.len,
            flag: BufFlag::Cache,
        };
        self.nodes.insert(
            node,
            RawCL {
                ptr: out.ptr.1,
                len: out.len,
            },
        );
        out
    }

    #[cfg(not(feature = "safe"))]
    pub fn get<T>(device: &CLDevice, len: usize) -> Buffer<T> {
        use crate::opencl::api::unified_ptr;

        /*assert!(
            !device.inner.borrow().ptrs.is_empty(),
            "no OpenCL allocations"
        );*/
        let node = Node::new(len);

        CL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);

            match buf_info_option {
                Some(buf_info) => {
                    let unified_ptr = if device.unified_mem() {
                        unified_ptr::<T>(device.queue(), buf_info.ptr, buf_info.len).unwrap()
                    } else {
                        std::ptr::null_mut()
                    };

                    Buffer {
                        ptr: (unified_ptr, buf_info.ptr, 0),
                        len: buf_info.len,
                        flag: BufFlag::Cache,
                    }
                }
                None => cache.add_node(device, node),
            }
        })
    }

    #[cfg(feature = "safe")]
    pub fn get<T>(device: &CLDevice, len: usize) -> Buffer<T> {
        Buffer::new(device, len)
    }

    pub(crate) fn arg_kernel_cache<T: 'static>(
        &mut self,
        device: &CLDevice,
        buffers: &[(&Buffer<T>, usize)],
        numbers: &[(T, usize)],
        output: Option<&Buffer<T>>,
        src: String,
    ) -> Result<Kernel, Error> {
        let type_ids = vec![TypeId::of::<T>(); numbers.len()];

        let mems: Vec<OclPtr> = buffers
            .iter()
            .map(|matrix| OclPtr(matrix.0.ptr.1))
            .collect();

        let outputmem = output.map(|output| OclPtr(output.ptr.1));

        let cache = &mut self.arg_kernel_cache;
        let kernel = cache.get(&(mems.clone(), type_ids.clone(), outputmem, src.clone()));
        match kernel {
            Some(kernel) => Ok(*kernel),
            None => {
                let program = create_program_with_source(&device.ctx(), &src)?;
                build_program(&program, &[device.device()], Some("-cl-std=CL1.2"))?; //-cl-single-precision-constant
                let kernel = create_kernels_in_program(&program)?[0];

                for (number, idx) in numbers {
                    set_kernel_arg(&kernel, *idx, number)?
                }

                for (buf, idx) in buffers {
                    set_kernel_arg(&kernel, *idx, &(buf.ptr.1))?;
                }

                if let Some(mem) = outputmem {
                    set_kernel_arg(&kernel, mems.len() + type_ids.len(), &mem)?;
                }

                cache.insert((mems, type_ids, outputmem, src), kernel);
                Ok(kernel)
            }
        }
    }

    pub fn arg_kernel_cache1(&mut self, device: &CLDevice, src: String) -> Result<Kernel, Error> {
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

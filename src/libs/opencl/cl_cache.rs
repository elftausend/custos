use std::{any::TypeId, collections::HashMap, ffi::c_void, cell::RefCell};
use crate::{Error, CDatatype, Node};
use super::{api::{build_program, create_kernels_in_program, create_program_with_source, Kernel, set_kernel_arg}, cl_device::InternCLDevice, PtrIdxSize, PtrIdxLen};

#[cfg(feature="opencl")]
use crate::Buffer;

thread_local! {
    pub static CL_CACHE: RefCell<CLCache> = RefCell::new(CLCache { 
        nodes: HashMap::new(), 
        arg_kernel_cache: HashMap::new(), 
        kernel_cache: HashMap::new() 
    })
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OclPtr(pub *mut c_void);

unsafe impl Send for OclPtr {}
unsafe impl Sync for OclPtr {}

type RawInfo = (OclPtr, usize);
type KernelIdent = (Vec<OclPtr>, Vec<TypeId>, Option<OclPtr>, String);
type KernelIdent1 = (Vec<OclPtr>, Option<OclPtr>, String);

#[derive(Debug)]
/// Stores kernels and outputs
pub struct CLCache {
    // TODO: Instead of a hashmap: vec?
    pub nodes: HashMap<Node, RawInfo>,
    pub(crate) arg_kernel_cache: HashMap<KernelIdent, Kernel>,
    pub(crate) kernel_cache: HashMap<KernelIdent1, Kernel>,
}

impl CLCache {
    pub fn add_node<T>(&mut self, device: InternCLDevice, node: Node) -> Buffer<T> {
        let out = Buffer::new(&device, node.len);
        self.nodes.insert(node, ( OclPtr(out.ptr.1), out.len ));
        out
    }

    #[cfg(not(feature="safe"))]
    pub fn get<T>(device: InternCLDevice, len: usize) -> Buffer<T> {
        use crate::opencl::api::unified_ptr;
        assert!(!device.cl.borrow().ptrs.is_empty(), "no OpenCL allocations");
        let node = Node::new(len);

        CL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let buf_info_option = cache.nodes.get(&node);
    
            match buf_info_option {
                Some(buf_info) => {
                    let unified_ptr = if device.unified_mem() {
                        unified_ptr::<T>(device.queue(), buf_info.0.0, buf_info.1).unwrap()
                    } else {
                        std::ptr::null_mut()
                    };

                    Buffer {
                        ptr: (unified_ptr, buf_info.0.0, 0),
                        len: buf_info.1
                    }
                }
                None => cache.add_node(device, node)
            }
        })
    }

    #[cfg(feature="safe")]
    pub fn get<T>(device: InternCLDevice, len: usize) -> Buffer<T> {
        Buffer::new(&device, len)
    }

    pub(crate) fn arg_kernel_cache<T: CDatatype>(&mut self, device: InternCLDevice, buffers: &[(&Buffer<T>, usize)], numbers: &[(T, usize)], output: Option<&Buffer<T>>, src: String) -> Result<Kernel, Error> {
        let type_ids = vec![TypeId::of::<T>(); numbers.len()];
        
        let mems: Vec<OclPtr> = buffers.iter()
            .map(|matrix| OclPtr(matrix.0.ptr.1))
            .collect();

        let outputmem = output.map(|output| OclPtr(output.ptr.1));
        
        let cache = &mut self.arg_kernel_cache;
        let kernel = cache.get(&(mems.clone(), type_ids.clone(), outputmem, src.clone()));
        match kernel { 
            Some(kernel) => Ok(kernel.clone()),
            None => {    
                let program = create_program_with_source(&device.ctx(), &src)?;
                build_program(&program, &[device.device()], Some("-cl-std=CL1.2"))?; //-cl-single-precision-constant
                let kernel = &create_kernels_in_program(&program)?[0];
                
                for (number, idx) in numbers {
                    set_kernel_arg(kernel, *idx, number)?
                }

                for (buf, idx) in buffers {
                    set_kernel_arg(kernel, *idx, &(buf.ptr.1))?;
                }

                if let Some(mem) = outputmem {
                    set_kernel_arg(kernel, mems.len()+type_ids.len(), &mem)?;
                }

                cache.insert((mems, type_ids, outputmem, src), kernel.clone());
                Ok(kernel.clone())
            },
        }
        
    }

    pub(crate) fn arg_kernel_cache1<T: CDatatype>(&mut self, device: InternCLDevice, buffers: &[PtrIdxLen], numbers: &[PtrIdxSize], output: Option<&Buffer<T>>, src: String) -> Result<Kernel, Error> {        
        let mems: Vec<OclPtr> = buffers.iter()
            .map(|ptrs| OclPtr(ptrs.0))
            .collect();

        let output = output.map(|output| OclPtr(output.ptr.1));
        
        let kernel = self.kernel_cache.get(&(mems.clone(), output, src.clone()));

        if let Some(kernel) = kernel {
            return Ok(*kernel);
        }

        let program = create_program_with_source(&device.ctx(), &src)?;
        build_program(&program, &[device.device()], Some("-cl-std=CL1.2"))?; //-cl-single-precision-constant
        let kernel = create_kernels_in_program(&program)?[0];

        for (buf, idx, _) in buffers {
            set_kernel_arg(&kernel, *idx, buf)?;
        }

        if let Some(mem) = output {
            set_kernel_arg(&kernel, buffers.len()+numbers.len(), &mem)?;
        }
        self.kernel_cache.insert((mems, output, src), kernel);
        Ok(kernel)
        
    }
}
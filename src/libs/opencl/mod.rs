pub use cl_cache::*;
pub use cl_device::*;
pub use cl_devices::*;
pub use kernel_options::*;
pub use ops::*;

pub mod api;
pub mod ops;
pub mod cl_device;
pub mod cl_devices;
mod kernel_options;
mod cl_cache;

use crate::{Matrix, GenericOCL, CPU, Node, Buffer, VecRead};
use self::api::{create_buffer, MemFlags, release_mem_object};

pub fn cpu_exec<T, F>(device: &InternCLDevice, matrix: &Matrix<T>, f: F) -> crate::Result<Matrix<T>> 
where 
    F: Fn(&crate::InternCPU, Matrix<T>) -> Matrix<T>,
    T: GenericOCL
{
    let cpu = CPU::new();

    if device.unified_mem() && !cfg!(feature="safe"){

        // host ptr matrix
        let no_drop = f(&cpu, *matrix);

        // use the host pointer to create an OpenCL buffer
        let cl_ptr = create_buffer(
            &device.ctx(), 
            MemFlags::MemReadWrite | MemFlags::MemUseHostPtr,
            no_drop.size(), 
            Some(&no_drop)
        )?;

        let old_ptr = CL_CACHE.with(|cache| {
            // add created buffer to the "caching chain"
            cache.borrow_mut().nodes.insert(Node::new(no_drop.size()), (OclPtr(cl_ptr), no_drop.size()))
        });

        // this pointer was overwritten, hence it can be deallocated
        if let Some(old) = old_ptr {
            unsafe {
                release_mem_object(old.0.0)?;
            }
        }
        
        // TODO: When should the buffer be freed, if the "safe" feature is used?

        // Both lines prevent the deallocation of the underlying buffer.
        //Box::into_raw(Box::new(no_drop)); // "safe" mode
        // TODO: Deallocate cpu buffer? This may leak memory.
        cpu.cpu.borrow_mut().ptrs.clear(); // default mode
        
        let buf = Buffer {
            ptr: (no_drop.ptr.0, cl_ptr),
            len: no_drop.size(),
        };
        return Ok(Matrix::from((buf, no_drop.dims())));
    }
    
    let x = if device.unified_mem() {
        matrix.clone()
    } else {
        // Read buffer that is allocated on an OpenCL device and create a new cpu matrix.
        Matrix::from((&cpu, matrix.dims(), device.read(matrix.as_buf())))
    };
    
    Ok(Matrix::from((device, f(&cpu, x))))
}

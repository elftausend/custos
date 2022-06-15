use std::ffi::c_void;

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

use crate::{Matrix, CDatatype, CPU, Node, Buffer, VecRead, number::Number};
use self::api::{create_buffer, MemFlags, release_mem_object};

pub fn to_unified<T>(device: &InternCLDevice, no_drop: Matrix<T>) -> crate::Result<*mut c_void> {
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

    // this pointer was overwritten previously, hence it can be deallocated
    if let Some(old) = old_ptr {
        unsafe {
            release_mem_object(old.0.0)?;
        }
    };

    Ok(cl_ptr)
}

pub fn construct_buffer<T>(device: &InternCLDevice, cpu: &crate::InternCPU, no_drop: Matrix<T>) -> crate::Result<Matrix<T>>  {
    let (host_ptr, no_drop_dims) = (no_drop.ptr.0, no_drop.dims());

    let cl_ptr = to_unified(device, no_drop)?;
    // TODO: When should the buffer be freed, if the "safe" feature is used?

    // Both lines prevent the deallocation of the underlying buffer.
    //Box::into_raw(Box::new(no_drop)); // "safe" mode
    // TODO: Deallocate cpu buffer? This may leak memory.
    cpu.cpu.borrow_mut().ptrs.clear(); // default mode
    
    let buf = Buffer {
        ptr: (host_ptr, cl_ptr, 0),
        len: no_drop_dims.0 * no_drop_dims.1,
    };
    Ok(Matrix::from((buf, no_drop_dims)))
}

pub fn cpu_exec<T, F>(device: &InternCLDevice, matrix: &Matrix<T>, f: F) -> crate::Result<Matrix<T>> 
where 
    F: Fn(&crate::InternCPU, Matrix<T>) -> Matrix<T>,
    T: CDatatype
{
    let cpu = CPU::new();

    if device.unified_mem() && !cfg!(feature="safe") { 
        // host ptr matrix
        let no_drop = f(&cpu, matrix.clone());
        return construct_buffer(device, &cpu, no_drop);
    }
    
    let x = if device.unified_mem() {
        matrix.clone()
    } else {
        // convert an OpenCL buffer to a cpu buffer
        Matrix::from((&cpu, matrix.dims(), device.read(matrix.as_buf())))
    };
    
    Ok(Matrix::from((device, f(&cpu, x))))
}

pub fn cpu_exec_lhs_rhs<T, F>(device: &InternCLDevice, lhs: &Matrix<T>, rhs: &Matrix<T>, f: F) -> crate::Result<Matrix<T>> 
where 
    F: Fn(&crate::InternCPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>,
    T: CDatatype
{
    let cpu = CPU::new();

    if device.unified_mem() && !cfg!(feature="safe") { 
        
        let no_drop = f(&cpu, lhs, rhs);
        return construct_buffer(device, &cpu, no_drop);
    }

    let (lhs, rhs) = if device.unified_mem() {
        (lhs.clone(), rhs.clone())
    } else {
        // convert an OpenCL buffer to a cpu buffer
        (
            Matrix::from((&cpu, lhs.dims(), device.read(lhs.as_buf()))),
            Matrix::from((&cpu, rhs.dims(), device.read(rhs.as_buf())))
        )
    };

    Ok(Matrix::from((device, f(&cpu, &lhs, &rhs))))
}

pub fn cpu_exec_scalar<T, F>(device: &InternCLDevice, matrix: &Matrix<T>, f: F) -> T 
where 
    F: Fn(&crate::InternCPU, Matrix<T>) -> T,
    T: Number
{
    let cpu = CPU::new();
    let x = if device.unified_mem() {
        matrix.clone()
    } else {
        // convert an OpenCL buffer to a cpu buffer
        Matrix::from((&cpu, matrix.dims(), device.read(matrix.as_buf())))
    };
    f(&cpu, x)

}
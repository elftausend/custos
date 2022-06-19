use std::ffi::c_void;
use crate::{Buffer, InternCudaDevice, number::Number};

use super::{fn_cache, api::{culaunch_kernel, cuOccupancyMaxPotentialBlockSize}};

pub trait AsCudaCvoidPtr {
    fn as_cvoid_ptr(&self) -> *mut c_void;
}

impl<T> AsCudaCvoidPtr for &Buffer<T> {
    fn as_cvoid_ptr(&self) -> *mut c_void {
        self.ptr.2 as *const u64 as *mut c_void
    }
}

impl<T> AsCudaCvoidPtr for Buffer<T> {
    fn as_cvoid_ptr(&self) -> *mut c_void {
        self.ptr.2 as *const u64 as *mut c_void
    }
}

impl<T: Number> AsCudaCvoidPtr for T {
    fn as_cvoid_ptr(&self) -> *mut c_void {
        self as *const T as *mut c_void
    }
}

/// uses calculated occupancy as launch configuration to launch a CUDA kernel
pub fn launch_kernel1d(len: usize, device: &InternCudaDevice, src: &str, fn_name: &str, params: &[&dyn AsCudaCvoidPtr]) -> crate::Result<()> {
    let params = params.iter()
        .map(|param| param.as_cvoid_ptr())
        .collect::<Vec<_>>();

    println!("params: {params:?}");
    
    let func = fn_cache(device, src, fn_name)?;

    let mut min_grid_size = 0;
    let mut block_size = 0;
    
    unsafe {cuOccupancyMaxPotentialBlockSize(
        &mut min_grid_size, &mut block_size, 
        func.0, 
        0, 0, 
        len as i32).to_result()?
    };
    let grid_size = (len as i32 + block_size - 1) / block_size;
    
    println!("grid_size: {grid_size}");
    println!("block_size: {block_size}");

    culaunch_kernel(
        &func, [grid_size as u32, 1, 1], [block_size as u32, 1, 1], 
        &device.stream(), &params
    )?;
    Ok(())
}

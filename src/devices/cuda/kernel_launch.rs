use crate::{number::Number, Buffer, OnDropBuffer, CUDA};
use std::{collections::HashMap, ffi::c_void};

use super::{
    api::{cuOccupancyMaxPotentialBlockSize, culaunch_kernel, FnHandle, Module, Stream},
    fn_cache, CUDAPtr, CUKernelCache, CudaSource,
};

/// Converts `Self` to a (cuda) *mut c_void.
/// This enables taking `Buffer` and a number `T` as an argument to an CUDA kernel.
/// # Example
/// ```
/// use custos::{CUDA, Buffer, cuda::AsCudaCvoidPtr, Base};
///
/// fn args(args: &[&dyn AsCudaCvoidPtr]) {
///     // ...
/// }
///
/// fn main() -> custos::Result<()> {
///     let device = CUDA::<Base>::new(0)?;
///
///     let buf = Buffer::<f32, _>::new(&device, 10);
///     let num = 4;
///     args(&[&num, &buf]);
///     Ok(())
/// }
/// ```
pub trait AsCudaCvoidPtr {
    /// Converts `Self` to a (cuda) *mut c_void.
    /// # Example
    /// ```
    /// use custos::{CUDA, Buffer, cuda::AsCudaCvoidPtr, Base};
    ///
    /// fn main() -> custos::Result<()> {
    ///     let device = CUDA::<Base>::new(0)?;
    ///     let buf = Buffer::<f32, _>::new(&device, 10);
    ///     
    ///     let _ptr = buf.as_cvoid_ptr();
    ///     Ok(())
    /// }
    ///
    fn as_cvoid_ptr(&self) -> *mut c_void;
}

impl<'a, T, Mods: OnDropBuffer> AsCudaCvoidPtr for &Buffer<'a, T, CUDA<Mods>> {
    #[inline]
    fn as_cvoid_ptr(&self) -> *mut c_void {
        &self.data.ptr as *const u64 as *mut c_void
    }
}

impl<'a, T, Mods: OnDropBuffer> AsCudaCvoidPtr for Buffer<'a, T, CUDA<Mods>> {
    #[inline]
    fn as_cvoid_ptr(&self) -> *mut c_void {
        &self.data.ptr as *const u64 as *mut c_void
    }
}

impl<T> AsCudaCvoidPtr for CUDAPtr<T> {
    #[inline]
    fn as_cvoid_ptr(&self) -> *mut c_void {
        &self.ptr as *const u64 as *mut c_void
    }
}

impl<T: Number> AsCudaCvoidPtr for T {
    #[inline]
    fn as_cvoid_ptr(&self) -> *mut c_void {
        self as *const T as *mut c_void
    }
}

/// Launch a CUDA kernel with the given grid and block sizes.
pub fn launch_kernel<Mods>(
    device: &CUDA<Mods>,
    grid: [u32; 3],
    blocks: [u32; 3],
    shared_mem_bytes: u32,
    src: impl CudaSource,
    fn_name: &str,
    params: &[&dyn AsCudaCvoidPtr],
) -> crate::Result<()> {
    let func = fn_cache(device, src, fn_name)?;
    launch_kernel_with_fn(
        device.stream(),
        &func,
        grid,
        blocks,
        shared_mem_bytes,
        params,
    )
}

/// Launch a CUDA kernel with the given CUDA function grid and block sizes.
pub fn launch_kernel_with_fn(
    stream: &Stream,
    func: &FnHandle,
    grid: [u32; 3],
    blocks: [u32; 3],
    shared_mem_bytes: u32,
    params: &[&dyn AsCudaCvoidPtr],
) -> crate::Result<()> {
    let params = params
        .iter()
        .map(|param| param.as_cvoid_ptr())
        .collect::<Vec<_>>();

    culaunch_kernel(func, grid, blocks, shared_mem_bytes, stream, &params)?;
    Ok(())
}

/// uses calculated occupancy as launch configuration to launch a CUDA kernel
/// # Safety
/// All kernel arguments must be set.
pub fn launch_kernel1d(
    len: usize,
    kernel_cache: &mut CUKernelCache,
    modules: &mut HashMap<FnHandle, Module>,
    stream: &Stream,
    src: impl CudaSource,
    fn_name: &str,
    params: &[&dyn AsCudaCvoidPtr],
) -> crate::Result<()> {
    let params = params
        .iter()
        .map(|param| param.as_cvoid_ptr())
        .collect::<Vec<_>>();

    let func = kernel_cache.kernel(modules, src, fn_name)?;

    let mut min_grid_size = 0;
    let mut block_size = 0;

    unsafe {
        cuOccupancyMaxPotentialBlockSize(
            &mut min_grid_size,
            &mut block_size,
            func.0,
            0,
            0,
            len as i32,
        )
        .to_result()?
    };
    let grid_size = (len as i32 + block_size - 1) / block_size;

    culaunch_kernel(
        &func,
        [grid_size as u32, 1, 1],
        [block_size as u32, 1, 1],
        0,
        stream,
        &params,
    )?;
    Ok(())
}

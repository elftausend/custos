use std::{ptr::null_mut, ffi::{c_void, CString}};

use crate::CUdeviceptr;

use super::{ffi::cuMemAlloc_v2, error::{CudaResult, CudaErrorKind}, cuInit, CUcontext, CUdevice, cuDeviceGet, cuCtxCreate_v2, cuMemFree_v2, cuDeviceGetCount, cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, cuModuleLoad, CUmodule, CUfunction, cuModuleGetFunction, cuLaunchKernel, CUstream, cuStreamCreate, cuStreamSynchronize, cuModuleLoadData};

pub fn cinit(flags: u32) -> CudaResult<()> {
    unsafe { cuInit(flags).into() }
}

#[derive(Debug)]
pub struct CudaIntDevice(CUdevice);

pub fn device_count() -> CudaResult<i32> {
    let mut count = 0;
    unsafe { cuDeviceGetCount(&mut count as *mut i32) }.to_result()?;
    Ok(count)
}

// TODO: cuda set device
pub fn device(ordinal: i32) -> CudaResult<CudaIntDevice> {
    if ordinal >= device_count()? {
        return Err(CudaErrorKind::InvalidDeviceIdx)
    } 

    let mut device = CudaIntDevice(0);
    unsafe { cuDeviceGet(&mut device.0 as *mut i32, ordinal) };
    Ok(device)
}

#[derive(Debug)]
pub struct Context(pub CUcontext);

pub fn create_context(device: &CudaIntDevice) -> CudaResult<Context> {
    let mut context = Context(null_mut());
    unsafe {
        // TODO: Flags
        cuCtxCreate_v2(&mut context.0 as *mut CUcontext, 0 , device.0).to_result()?;
    }
    Ok(context)
}

pub fn cumalloc<T>(len: usize) -> CudaResult<CUdeviceptr> {
    let bytes = len * core::mem::size_of::<T>();

    if bytes == 0 {
        return Err(CudaErrorKind::InvalidAllocSize)
    }

    let mut ptr: CUdeviceptr = 0;
    unsafe { cuMemAlloc_v2(&mut ptr , bytes).to_result()? };
    Ok(ptr)
}

pub unsafe fn cufree(ptr: CUdeviceptr) -> CudaResult<()> {
    cuMemFree_v2(ptr).into()
}

pub fn cuwrite<T>(dst: CUdeviceptr, src_host: &[T]) -> CudaResult<()> {
    let bytes_to_copy = src_host.len() * std::mem::size_of::<T>();
    unsafe { cuMemcpyHtoD_v2(
        dst, 
        src_host.as_ptr() as *const c_void, 
        bytes_to_copy) 
    }.into()
}

pub fn curead<T>(dst_host: &mut [T], src: CUdeviceptr,) -> CudaResult<()> {
    let bytes_to_copy = dst_host.len() * std::mem::size_of::<T>();
    unsafe { cuMemcpyDtoH_v2(
        dst_host.as_mut_ptr() as *mut c_void, 
        src, 
        bytes_to_copy) 
    }.into()
    
}

#[derive(Debug)]
pub struct Module(pub CUmodule);

pub fn load_module(fname: &str) -> CudaResult<Module> {
    let fname = CString::new(fname).unwrap();
    
    let mut module = Module(null_mut());
    unsafe { cuModuleLoad(&mut module.0, fname.as_ptr()) }.to_result()?;
    Ok(module)
}

pub fn load_module_data(src: CString) -> CudaResult<Module> {
    let mut module = Module(null_mut());
    unsafe { cuModuleLoadData(&mut module.0, src.as_ptr() as *const c_void) }.to_result()?;
    Ok(module)
}

#[derive(Debug)]
pub struct FnHandle(pub CUfunction);

pub fn module_get_fn(module: Module, fn_name: &str) -> CudaResult<FnHandle> {
    let fn_name = CString::new(fn_name).unwrap();

    let mut handle = FnHandle(null_mut());
    unsafe { cuModuleGetFunction(&mut handle.0, module.0, fn_name.as_ptr()) }.to_result()?;
    Ok(handle)
}

#[derive(Debug)]
pub struct Stream(pub CUstream);

impl Stream {
    pub fn sync(&self) -> CudaResult<()> {
        unsafe { cuStreamSynchronize(self.0) }.to_result()
    }
} 

pub fn create_stream() -> CudaResult<Stream> {
    let mut ph_stream = Stream(null_mut());
    unsafe { cuStreamCreate(&mut ph_stream.0, 0) }.to_result()?;
    Ok(ph_stream)
}

pub fn launch_kernel(f: &FnHandle, gws: [u32; 3], lws: [u32; 3], stream: &mut Stream, params: &[*mut c_void]) -> CudaResult<()> {
    unsafe { cuLaunchKernel(
        f.0, gws[0], 
        gws[1], gws[2], 
        lws[0], lws[1], 
        lws[2], 0, 
        stream.0, params.as_ptr() as *mut _, std::ptr::null_mut()
    )}.to_result()
}
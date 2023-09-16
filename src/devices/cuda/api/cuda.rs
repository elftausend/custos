use super::{
    cuCtxCreate_v2, cuCtxDestroy, cuDeviceGet, cuDeviceGetCount, cuGraphDestroy,
    cuGraphExecDestroy, cuInit, cuLaunchKernel, cuMemFree_v2, cuMemcpyDtoHAsync_v2,
    cuMemcpyDtoH_v2, cuMemcpyHtoDAsync_v2, cuMemcpyHtoD_v2, cuModuleGetFunction, cuModuleLoad,
    cuModuleLoadData, cuModuleUnload, cuStreamCreate, cuStreamIsCapturing, cuStreamSynchronize,
    error::{CudaErrorKind, CudaResult},
    ffi::cuMemAlloc_v2,
    CUcontext, CUdevice, CUfunction, CUgraphExec_st, CUgraph_st, CUmodule, CUstream,
    CUstreamCaptureStatus,
};

use core::ptr::NonNull;
use std::{
    ffi::{c_void, CString},
    ptr::null_mut,
};

pub fn cuinit(flags: u32) -> CudaResult<()> {
    unsafe { cuInit(flags).into() }
}

pub type CUdeviceptr = core::ffi::c_ulonglong;

#[derive(Debug)]
pub struct CudaIntDevice(pub CUdevice);

pub fn device_count() -> CudaResult<i32> {
    let mut count = 0;
    unsafe { cuDeviceGetCount(&mut count as *mut i32) }.to_result()?;
    Ok(count)
}

// TODO: cuda set device
pub fn device(ordinal: i32) -> CudaResult<CudaIntDevice> {
    if ordinal >= device_count()? {
        return Err(CudaErrorKind::InvalidDeviceIdx);
    }

    let mut device = CudaIntDevice(0);
    unsafe { cuDeviceGet(&mut device.0 as *mut i32, ordinal) };
    Ok(device)
}

#[derive(Debug)]
pub struct Context(pub CUcontext);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            cuCtxDestroy(self.0);
        }
    }
}

pub fn create_context(device: &CudaIntDevice) -> CudaResult<Context> {
    let mut context = Context(null_mut());
    unsafe {
        // TODO: Flags
        cuCtxCreate_v2(&mut context.0 as *mut CUcontext, 0, device.0).to_result()?;
    }
    Ok(context)
}

pub fn cumalloc<T>(len: usize) -> CudaResult<CUdeviceptr> {
    let bytes = len * core::mem::size_of::<T>();

    if bytes == 0 {
        return Err(CudaErrorKind::InvalidAllocSize);
    }

    let mut ptr: CUdeviceptr = 0;
    unsafe { cuMemAlloc_v2(&mut ptr, bytes).to_result()? };
    Ok(ptr)
}

/// Free CUDA GPU memory
/// # Safety
/// FFI, `ptr` must be a valid pointer.
pub unsafe fn cufree(ptr: CUdeviceptr) -> CudaResult<()> {
    cuMemFree_v2(ptr).into()
}

pub fn cu_write<T>(dst: CUdeviceptr, src_host: &[T]) -> CudaResult<()> {
    let bytes_to_copy = std::mem::size_of_val(src_host);
    unsafe { cuMemcpyHtoD_v2(dst, src_host.as_ptr() as *const c_void, bytes_to_copy) }.into()
}

pub fn cu_write_async<T>(dst: CUdeviceptr, src_host: &[T], stream: &Stream) -> CudaResult<()> {
    let bytes_to_copy = std::mem::size_of_val(src_host);
    unsafe {
        cuMemcpyHtoDAsync_v2(
            dst,
            src_host.as_ptr() as *const c_void,
            bytes_to_copy,
            stream.0,
        )
    }
    .into()
}

pub fn cu_read<T>(dst_host: &mut [T], src: CUdeviceptr) -> CudaResult<()> {
    let bytes_to_copy = std::mem::size_of_val(dst_host);
    unsafe { cuMemcpyDtoH_v2(dst_host.as_mut_ptr() as *mut c_void, src, bytes_to_copy) }.into()
}

pub fn cu_read_async<T>(dst_host: &mut [T], src: CUdeviceptr, stream: &Stream) -> CudaResult<()> {
    let bytes_to_copy = std::mem::size_of_val(dst_host);
    unsafe {
        cuMemcpyDtoHAsync_v2(
            dst_host.as_mut_ptr() as *mut c_void,
            src,
            bytes_to_copy,
            stream.0,
        )
    }
    .into()
}

#[derive(Debug)]
pub struct Module(pub CUmodule);

impl Module {
    pub fn function(&self, fn_name: &str) -> CudaResult<FnHandle> {
        module_get_fn(self, fn_name)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.0).to_result().unwrap();
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FnHandle(pub CUfunction);

pub fn module_get_fn(module: &Module, fn_name: &str) -> CudaResult<FnHandle> {
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

    pub fn capture_status(&self) -> CudaResult<CUstreamCaptureStatus> {
        let mut capture_status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe { cuStreamIsCapturing(self.0, &mut capture_status).to_result()? };
        Ok(capture_status)
    }
}

pub fn create_stream() -> CudaResult<Stream> {
    let mut ph_stream = Stream(null_mut());
    unsafe { cuStreamCreate(&mut ph_stream.0, 0) }.to_result()?;
    Ok(ph_stream)
}

pub fn culaunch_kernel(
    f: &FnHandle,
    grid: [u32; 3],
    blocks: [u32; 3],
    shared_mem_bytes: u32,
    stream: &Stream,
    params: &[*mut c_void],
) -> CudaResult<()> {
    unsafe {
        cuLaunchKernel(
            f.0,
            grid[0],
            grid[1],
            grid[2],
            blocks[0],
            blocks[1],
            blocks[2],
            shared_mem_bytes,
            stream.0,
            params.as_ptr() as *mut _,
            std::ptr::null_mut(),
        )
    }
    .to_result()?;

    // TODO: sync here or elsewhere?
    //stream.sync()?;

    //    unsafe {cuCtxSynchronize().to_result()?};

    Ok(())
}

pub struct Graph(pub NonNull<CUgraph_st>);

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { cuGraphDestroy(self.0.as_ptr()) }
            .to_result()
            .unwrap()
    }
}

pub struct GraphExec(pub NonNull<CUgraphExec_st>);

impl Drop for GraphExec {
    fn drop(&mut self) {
        unsafe { cuGraphExecDestroy(self.0.as_ptr()) }
            .to_result()
            .unwrap()
    }
}

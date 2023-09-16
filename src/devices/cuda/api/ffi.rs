#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::{ffi::c_void, os::raw::c_char};

use super::{
    error::{CudaErrorKind, CudaResult},
    CUdeviceptr,
};
pub type CUdevice = std::os::raw::c_int;

pub enum CUctx_st {}
pub type CUcontext = *mut CUctx_st;

pub enum CUmod_st {}
pub type CUmodule = *mut CUmod_st;

pub enum CUfunc_st {}
pub type CUfunction = *mut CUfunc_st;

pub enum CUstream_st {}
pub type CUstream = *mut CUstream_st;

pub enum CUgraph_st {}
pub type CUgraph = *mut CUgraph_st;

pub enum CUgraphExec_st {}
pub type CUgraphExec = *mut CUgraphExec_st;

#[repr(u32)]
pub enum CUStreamCaptureMode {
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED = 2,
}

#[repr(u32)]
pub enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_UNKNOWN = 999,
}

#[derive(Debug, PartialEq, Eq)]
#[repr(C)]
pub enum CUstreamCaptureStatus {
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
}

#[repr(C)]
pub enum CUgraphInstantiate_flags {
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
    CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2,
    CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4,
    CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
}

impl From<CUresult> for CudaResult<()> {
    fn from(result: CUresult) -> Self {
        match result {
            CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(CudaErrorKind::from(result as u32)),
        }
    }
}

impl CUresult {
    pub fn to_result(self) -> CudaResult<()> {
        match self {
            CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(CudaErrorKind::from(self as u32)),
        }
    }
}

#[link(name = "cuda")]
extern "C" {
    pub fn cuInit(flags: u32) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut i32) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    pub fn cuDeviceGetAttribute(
        pi: *mut i32,
        attrib: CUdevice_attribute,
        device: CUdevice,
    ) -> CUresult;
    pub fn cuCtxCreate_v2(context: *mut CUcontext, flags: u32, device: CUdevice) -> CUresult;
    pub fn cuCtxDestroy(context: CUcontext);
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuMemAlloc_v2(ptr: *mut CUdeviceptr, size: usize) -> CUresult;
    pub fn cuMemFree_v2(ptr: CUdeviceptr) -> CUresult;

    pub fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CUresult;

    pub fn cuMemcpyHtoD_v2(
        dst_device: CUdeviceptr,
        src_host: *const c_void,
        bytes_to_copy: usize,
    ) -> CUresult;

    pub fn cuMemcpyHtoDAsync_v2(
        dst_device: CUdeviceptr,
        src_host: *const c_void,
        bytes_to_copy: usize,
        stream: CUstream,
    ) -> CUresult;

    pub fn cuMemcpyDtoH_v2(
        dst_host: *mut c_void,
        src_device: CUdeviceptr,
        bytes_to_copy: usize,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(
        dst_host: *mut c_void,
        src_device: CUdeviceptr,
        bytes_to_copy: usize,
        stream: CUstream,
    ) -> CUresult;

    pub fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, data: *const c_void) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        module: CUmodule,
        fn_name: *const c_char,
    ) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuStreamCreate(ph_stream: *mut CUstream, flags: u32) -> CUresult;
    pub fn cuStreamDestroy(hstream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    pub fn cuOccupancyMaxPotentialBlockSize(
        min_grid_size: *mut i32,
        block_size: *mut i32,
        func: CUfunction,
        block_size_to_dyn_b2d_size: usize,
        dyn_smem_size: usize,
        block_size_limit: i32,
    ) -> CUresult;

    pub fn cuStreamBeginCapture(stream: CUstream, capture_mode: CUStreamCaptureMode) -> CUresult;
    pub fn cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph) -> CUresult;
    pub fn cuStreamIsCapturing(
        stream: CUstream,
        capture_status: *mut CUstreamCaptureStatus,
    ) -> CUresult;

    pub fn cuGraphDestroy(graph: CUgraph) -> CUresult;

    pub fn cuGraphInstantiate(
        graph_exec: *mut CUgraphExec,
        graph: CUgraph,
        flags: CUgraphInstantiate_flags,
    ) -> CUresult;

    pub fn cuGraphLaunch(graph_exec: CUgraphExec, stream: CUstream) -> CUresult;

    pub fn cuGraphExecDestroy(graph_exec: CUgraphExec) -> CUresult;

    // unified memory

}

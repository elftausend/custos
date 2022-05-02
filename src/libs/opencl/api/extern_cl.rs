#![allow(non_camel_case_types)]

use std::ffi::c_void;

pub type size_t = usize;
pub type c_char = i8;

pub type cl_platform_id     = *mut c_void;
pub type cl_device_id       = *mut c_void;
pub type cl_context         = *mut c_void;
pub type cl_command_queue   = *mut c_void;
pub type cl_mem             = *mut c_void;
pub type cl_program         = *mut c_void;
pub type cl_kernel          = *mut c_void;
pub type cl_event           = *mut c_void;

pub type cl_int                             = i32;
pub type cl_uint                            = u32;
pub type cl_long                            = i64;
pub type cl_ulong                           = u64;
//pub type cl_half                            = "u16" f16;
pub type cl_bool                            = cl_uint;
pub type cl_bitfield                        = cl_ulong;
pub type cl_device_type                     = cl_bitfield;
pub type cl_platform_info                   = cl_uint;
pub type cl_device_info                     = cl_uint;
pub type cl_command_queue_properties        = cl_bitfield;
pub type cl_context_properties              = isize;
pub type cl_mem_flags                       = cl_bitfield;
pub type cl_program_info                    = cl_uint;
pub type cl_program_build_info              = cl_uint;
pub type cl_map_flags                       = cl_bitfield;



 
#[cfg_attr(target_os = "macos", link(name = "OpenCL", kind = "framework"))]
#[cfg_attr(target_os = "windows", link(name = "OpenCL"))]
#[cfg_attr(not(target_os = "macos"), link(name = "OpenCL"))]
extern "system" {
    pub fn clGetPlatformIDs(num_entries: cl_uint,
                            platforms: *mut cl_platform_id,
                            num_platforms: *mut cl_uint) -> cl_int;
    
    pub fn clGetPlatformInfo(platform: cl_platform_id,
        param_name: cl_platform_info,
        param_value_size: size_t,
        param_value: *mut c_void,
        param_value_size_ret: *mut size_t) -> cl_int;

    pub fn clGetDeviceIDs(platform: cl_platform_id,
        device_type: cl_device_type,
        num_entries: cl_uint,
        devices: *mut cl_device_id,
        num_devices: *mut cl_uint) -> cl_int;
    
    pub fn clGetDeviceInfo(device: cl_device_id,
        param_name: cl_device_info,
        param_value_size: size_t,
        param_value: *mut c_void,
        param_value_size_ret: *mut size_t) -> cl_int;
    
    pub fn clCreateContext(properties: *const cl_context_properties,
        num_devices: cl_uint,
        devices: *const cl_device_id,
        pfn_notify: *mut c_void,
        user_data: *mut c_void,
        errcode_ret: *mut cl_int) -> cl_context;
    
    pub fn clReleaseContext(context: cl_context) -> cl_int;
    
    pub fn clCreateCommandQueue(context: cl_context,
        device: cl_device_id,
        properties: cl_command_queue_properties,
        errcode_ret: *mut cl_int) -> cl_command_queue;
    
    pub fn clFinish(command_queue: cl_command_queue) -> cl_int;
    
    pub fn clReleaseCommandQueue(command_queue: cl_command_queue) -> cl_int;
    
    pub fn clWaitForEvents(num_events: cl_uint,
        event_list: *const cl_event) -> cl_int;
    
    pub fn clReleaseEvent(event: cl_event) -> cl_int;
    
    pub fn clCreateBuffer(context: cl_context,
        flags: cl_mem_flags,
        size: size_t,
        host_ptr: *mut c_void,
        errcode_ret: *mut cl_int) -> cl_mem;
    
    pub fn clRetainMemObject(memobj: cl_mem) -> cl_int;

    pub fn clReleaseMemObject(memobj: cl_mem) -> cl_int;
    
    pub fn clEnqueueReadBuffer(command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_read: cl_bool,
        offset: size_t,
        cb: size_t,
        ptr: *mut c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;
    
    pub fn clEnqueueWriteBuffer(command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_write: cl_bool,
        offset: size_t,
        cb: size_t,
        ptr: *const c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;
    
    pub fn clEnqueueCopyBuffer(command_queue: cl_command_queue,
        src_buffer: cl_mem,
        dst_buffer: cl_mem,
        src_offset: size_t,
        dst_offset: size_t,
        cb: size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueMapBuffer(command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_map: cl_bool,
        map_flags: cl_map_flags,
        offset: size_t,
        size: size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event,
        errorcode_ret: *mut cl_int) -> *mut c_void;
    
    pub fn clEnqueueFillBuffer(command_queue: cl_command_queue,
        buffer: cl_mem,
        pattern: *const c_void,
        pattern_size: size_t,
        offset: size_t,
        size: size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;
    
    pub fn clReleaseProgram(program: cl_program) -> cl_int;
    
    pub fn clGetProgramInfo(program: cl_program,
        param_name: cl_program_info,
        param_value_size: size_t,
        param_value: *mut c_void,
        param_value_size_ret: *mut size_t) -> cl_int;
    
    pub fn clCreateProgramWithSource(context: cl_context,
        count: cl_uint,
        strings: *const *const c_char,
        lengths: *const size_t,
        errcode_ret: *mut cl_int) -> cl_program;
    
    pub fn clGetProgramBuildInfo(program: cl_program,
        device: cl_device_id,
        param_name: cl_program_build_info,
        param_value_size: size_t,
        param_value: *mut c_void,
        param_value_size_ret: *mut size_t) -> cl_int;
    
    pub fn clBuildProgram(program: cl_program,
        num_devices: cl_uint,
        device_list: *const cl_device_id,
        options: *const c_char,
        pfn_notify: *mut c_void,
        user_data: *mut c_void) -> cl_int;

    pub fn clCreateKernel(program: cl_program,
        kernel_name: *const c_char,
        errcode_ret: *mut cl_int) -> cl_kernel;
    
    pub fn clCreateKernelsInProgram(program: cl_program,
        num_kernels: cl_uint,
        kernels: *mut cl_kernel,
        num_kernels_ret: *mut cl_uint) -> cl_int;
    pub fn clReleaseKernel(kernel: cl_kernel) -> cl_int;

    pub fn clSetKernelArg(kernel: cl_kernel,
        arg_index: cl_uint,
        arg_size: size_t,
        arg_value: *const c_void) -> cl_int;

    pub fn clEnqueueNDRangeKernel(command_queue: cl_command_queue,
        kernel: cl_kernel,
        work_dim: cl_uint,
        global_work_offset: *const size_t,
        global_work_dims: *const size_t,
        local_work_dims: *const size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;
}

#![allow(dead_code)]

use std::{
    ffi::{c_void, CString},
    mem::size_of,
    rc::Rc,
    usize, vec,
};

use crate::Error;

use super::{ffi::*, OCLErrorKind};

#[derive(Clone, Copy, Debug)]
pub struct Platform(cl_platform_id);

impl Platform {
    pub fn as_ptr(self) -> *mut cl_platform_id {
        self.0 as *mut cl_platform_id
    }
}

pub fn get_platforms() -> Result<Vec<Platform>, Error> {
    let mut platforms: cl_uint = 0;
    let value = unsafe { clGetPlatformIDs(0, std::ptr::null_mut(), &mut platforms) };

    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    let mut vec: Vec<usize> = vec![0; platforms as usize];
    let (ptr, len, cap) = (vec.as_mut_ptr(), vec.len(), vec.capacity());

    let mut platforms_vec: Vec<Platform> = unsafe {
        core::mem::forget(vec);
        Vec::from_raw_parts(ptr as *mut Platform, len, cap)
    };

    let value = unsafe {
        clGetPlatformIDs(
            platforms,
            platforms_vec.as_mut_ptr() as *mut cl_platform_id,
            std::ptr::null_mut(),
        )
    };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(platforms_vec)
}

#[derive(Clone, Copy)]
pub enum PlatformInfo {
    PlatformName = 0x0903,
}
pub fn get_platform_info(platform: Platform, param_name: PlatformInfo) -> String {
    let mut size: size_t = 0;
    unsafe {
        clGetPlatformInfo(
            platform.0,
            param_name as cl_platform_info,
            0,
            std::ptr::null_mut(),
            &mut size,
        );
    };

    let mut param_value = vec![32u8; size];

    unsafe {
        clGetPlatformInfo(
            platform.0,
            param_name as cl_platform_info,
            size,
            param_value.as_mut_ptr() as *mut c_void,
            std::ptr::null_mut(),
        );
    };

    println!("param value: {:?}", param_value);
    String::from_utf8_lossy(&param_value).to_string()
}

pub enum DeviceType {
    DEFAULT = (1 << 0),
    CPU = (1 << 1),
    GPU = (1 << 2),
    ACCELERATOR = (1 << 3),
    //ALL =         0xFFFFFFFF
}

#[derive(Copy, Clone)]
pub enum DeviceInfo {
    MaxMemAllocSize = 0x1010,
    GlobalMemSize = 0x101F,
    NAME = 0x102B,
    VERSION = 0x102F,
    HostUnifiedMemory = 0x1035,
}
#[derive(Clone, Copy, Debug, Hash)]
pub struct CLIntDevice(pub cl_device_id);

impl CLIntDevice {
    pub fn get_name(self) -> Result<String, Error> {
        Ok(get_device_info(self, DeviceInfo::NAME)?.string)
    }
    pub fn get_version(self) -> Result<String, Error> {
        Ok(get_device_info(self, DeviceInfo::VERSION)?.string)
    }
    pub fn get_global_mem(self) -> Result<u64, Error> {
        Ok(get_device_info(self, DeviceInfo::GlobalMemSize)?.size)
    }
    pub fn get_max_mem_alloc(self) -> Result<u64, Error> {
        Ok(get_device_info(self, DeviceInfo::MaxMemAllocSize)?.size)
    }
    pub fn unified_mem(self) -> Result<bool, Error> {
        Ok(get_device_info(self, DeviceInfo::HostUnifiedMemory)?.size != 0)
    }
}

pub fn get_device_ids(platform: Platform, device_type: &u64) -> Result<Vec<CLIntDevice>, Error> {
    let mut num_devices: cl_uint = 0;
    let value = unsafe {
        clGetDeviceIDs(
            platform.0,
            *device_type,
            0,
            std::ptr::null_mut(),
            &mut num_devices,
        )
    };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }

    let mut vec: Vec<usize> = vec![0; num_devices as usize];
    let (ptr, len, cap) = (vec.as_mut_ptr(), vec.len(), vec.capacity());

    let mut devices: Vec<CLIntDevice> = unsafe {
        core::mem::forget(vec);
        Vec::from_raw_parts(ptr as *mut CLIntDevice, len, cap)
    };

    let value = unsafe {
        clGetDeviceIDs(
            platform.0,
            DeviceType::GPU as u64,
            num_devices,
            devices.as_mut_ptr() as *mut cl_device_id,
            std::ptr::null_mut(),
        )
    };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(devices)
}

pub struct DeviceReturnInfo {
    pub string: String,
    pub size: u64,
    pub data: Vec<u8>,
}

pub fn get_device_info(
    device: CLIntDevice,
    param_name: DeviceInfo,
) -> Result<DeviceReturnInfo, Error> {
    let mut size: size_t = 0;
    let value = unsafe {
        clGetDeviceInfo(
            device.0,
            param_name as cl_device_info,
            0,
            std::ptr::null_mut(),
            &mut size,
        )
    };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    let mut param_value = vec![0; size];
    let value = unsafe {
        clGetDeviceInfo(
            device.0,
            param_name as cl_device_info,
            size,
            param_value.as_mut_ptr() as *mut c_void,
            std::ptr::null_mut(),
        )
    };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    let string = String::from_utf8_lossy(&param_value).to_string();
    let size = param_value.iter().fold(0, |x, &i| x << 4 | i as u64);

    Ok(DeviceReturnInfo {
        string,
        size,
        data: param_value,
    })
}

// TODO: implement drop
#[derive(Debug, Hash, /*remove:*/ Clone)]
pub struct Context(pub cl_context);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { clReleaseContext(self.0) };
    }
}

pub fn create_context(devices: &[CLIntDevice]) -> Result<Context, Error> {
    let mut err = 0;
    let r = unsafe {
        clCreateContext(
            std::ptr::null(),
            devices.len() as u32,
            devices.as_ptr() as *const *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut err,
        )
    };
    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(Context(r))
}

#[derive(Debug /* remove: */, Clone)]
pub struct CommandQueue(pub cl_command_queue);

impl CommandQueue {
    pub fn release(&mut self) -> Result<(), Error> {
        let err = unsafe { clReleaseCommandQueue(self.0) };
        if err != 0 {
            return Err(OCLErrorKind::from_value(err).into());
        }
        Ok(())
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        self.release().unwrap();
    }
}

pub fn create_command_queue(context: &Context, device: CLIntDevice) -> Result<CommandQueue, Error> {
    let mut err = 0;
    let r = unsafe { clCreateCommandQueue(context.0, device.0, 0, &mut err) };

    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(CommandQueue(r))
}

pub fn finish(cq: CommandQueue) {
    unsafe { clFinish(cq.0) };
}

#[derive(Debug, Clone, Copy)]
pub struct Event(pub cl_event);

impl Event {
    pub fn wait(self) -> Result<(), Error> {
        wait_for_event(self)
    }
    pub fn release(self) {
        release_event(self).unwrap();
    }
}

pub fn wait_for_event(event: Event) -> Result<(), Error> {
    let event_vec: Vec<Event> = vec![event];

    let value = unsafe { clWaitForEvents(1, event_vec.as_ptr() as *mut cl_event) };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    event.release();
    Ok(())
}

pub fn release_event(event: Event) -> Result<(), Error> {
    let value = unsafe { clReleaseEvent(event.0) };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(())
}

pub enum MemFlags {
    MemReadWrite = 1,
    MemWriteOnly = 1 << 1,
    MemReadOnly = 1 << 2,
    MemUseHostPtr = 1 << 3,
    MemAllocHostPtr = 1 << 4,
    MemCopyHostPtr = 1 << 5,
    MemHostWriteOnly = 1 << 7,
    MemHostReadOnly = 1 << 8,
    MemHostNoAccess = 1 << 9,
}

impl core::ops::BitOr for MemFlags {
    type Output = u64;

    fn bitor(self, rhs: Self) -> Self::Output {
        self as u64 | rhs as u64
    }
}

pub fn create_buffer<T>(
    context: &Context,
    flag: u64,
    size: usize,
    data: Option<&[T]>,
) -> Result<*mut c_void, Error> {
    let mut err = 0;
    let host_ptr = match data {
        Some(d) => d.as_ptr() as cl_mem,
        None => std::ptr::null_mut(),
    };
    let r = unsafe {
        clCreateBuffer(
            context.0,
            flag as u64,
            size * core::mem::size_of::<T>(),
            host_ptr,
            &mut err,
        )
    };

    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(r)
}

/// # Safety
/// valid mem object
pub unsafe fn release_mem_object(ptr: *mut c_void) -> Result<(), Error> {
    let value = clReleaseMemObject(ptr);
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(())
}

pub(crate) fn retain_mem_object(mem: *mut c_void) -> Result<(), Error> {
    let value = unsafe { clRetainMemObject(mem) };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(())
}

/// # Safety
/// valid mem object
pub unsafe fn enqueue_write_buffer<T>(
    cq: &CommandQueue,
    mem: *mut c_void,
    data: &[T],
    block: bool,
) -> Result<Event, Error> {
    let mut events = vec![std::ptr::null_mut(); 1];

    let value = clEnqueueWriteBuffer(
        cq.0,
        mem,
        block as u32,
        0,
        data.len() * core::mem::size_of::<T>(),
        data.as_ptr() as *mut c_void,
        0,
        std::ptr::null(),
        events.as_mut_ptr() as *mut cl_event,
    );
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(Event(events[0]))
}

/// # Safety
/// valid mem object
pub unsafe fn enqueue_read_buffer<T>(
    cq: &CommandQueue,
    mem: *mut c_void,
    data: &mut [T],
    block: bool,
) -> Result<Event, Error> {
    let mut events = vec![std::ptr::null_mut(); 1];
    let value = clEnqueueReadBuffer(
        cq.0,
        mem,
        block as u32,
        0,
        data.len() * core::mem::size_of::<T>(),
        data.as_ptr() as *mut c_void,
        0,
        std::ptr::null(),
        events.as_mut_ptr() as *mut cl_event,
    );
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(Event(events[0]))
}
pub(crate) fn enqueue_full_copy_buffer<T>(
    cq: &CommandQueue,
    src_mem: *mut c_void,
    dst_mem: *mut c_void,
    size: usize,
) -> Result<(), Error> {
    let mut events = vec![std::ptr::null_mut(); 1];
    let value = unsafe {
        clEnqueueCopyBuffer(
            cq.0,
            src_mem,
            dst_mem,
            0,
            0,
            size * size_of::<T>(),
            0,
            std::ptr::null(),
            events.as_mut_ptr() as *mut cl_event,
        )
    };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    wait_for_event(Event(events[0]))
}

pub(crate) fn unified_ptr<T>(
    cq: &CommandQueue,
    ptr: *mut c_void,
    len: usize,
) -> Result<*mut T, Error> {
    unsafe { enqueue_map_buffer::<T>(cq, ptr, true, 2 | 1, 0, len).map(|ptr| ptr as *mut T) }
}

/// map_flags: Read: 1, Write: 2,
/// # Safety
/// valid mem object
pub unsafe fn enqueue_map_buffer<T>(
    cq: &CommandQueue,
    buffer: *mut c_void,
    block: bool,
    map_flags: u64,
    offset: usize,
    len: usize,
) -> Result<*mut c_void, Error> {
    let offset = offset * core::mem::size_of::<T>();
    let size = len * core::mem::size_of::<T>();

    let mut event = vec![std::ptr::null_mut(); 1];

    let mut err = 0;

    let ptr = clEnqueueMapBuffer(
        cq.0,
        buffer,
        block as u32,
        map_flags,
        offset,
        size,
        0,
        std::ptr::null(),
        event.as_mut_ptr() as *mut cl_event,
        &mut err,
    );

    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }

    let e = Event(event[0]);
    wait_for_event(e)?;
    Ok(ptr)
}
/*
pub fn enqueue_fill_buffer<T>(cq: &CommandQueue, mem: &Mem, pattern: Vec<T>) -> Event {
    let mut events = vec![std::ptr::null_mut();1];
    let offset = 0;
    let pattern_size = core::mem::size_of::<T>();
    let size = pattern_size*pattern.len();
    let err = unsafe {clEnqueueFillBuffer(cq.0, mem.0, pattern.as_ptr() as *mut c_void, pattern_size, offset, size, 0, std::ptr::null(), events.as_mut_ptr() as *mut cl_event)};
    println!("err enq copy bff: {}", err);
    Event(events[0])
}
*/
// TODO: implement drop
pub struct Program(pub cl_program);

impl Program {
    pub fn release(&mut self) {
        release_program(self).unwrap();
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        release_program(self).unwrap()
    }
}

enum ProgramInfo {
    BinarySizes = 0x1165,
    Binaries = 0x1166,
}

enum ProgramBuildInfo {
    Status = 0x1181,
    BuildLog = 0x1183,
}

pub fn release_program(program: &mut Program) -> Result<(), Error> {
    let value = unsafe { clReleaseProgram(program.0) };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(())
}

pub fn create_program_with_source(context: &Context, src: &str) -> Result<Program, Error> {
    let mut err = 0;
    let cs = CString::new(src).expect("No cstring for you!");
    let lens = vec![cs.as_bytes().len()];
    let cstring: Vec<*const _> = vec![cs.as_ptr()];
    let r = unsafe {
        clCreateProgramWithSource(
            context.0,
            1,
            cstring.as_ptr() as *const *const _,
            lens.as_ptr() as *const usize,
            &mut err,
        )
    };
    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(Program(r))
}

pub fn build_program(
    program: &Program,
    devices: &[CLIntDevice],
    options: Option<&str>,
) -> Result<(), Error> {
    let len = devices.len();

    let err = if let Some(options) = options {
        let options = CString::new(options).unwrap();
        unsafe {
            clBuildProgram(
                program.0,
                len as u32,
                devices.as_ptr() as *const *mut c_void,
                options.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        }
    } else {
        unsafe {
            clBuildProgram(
                program.0,
                len as u32,
                devices.as_ptr() as *const *mut c_void,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        }
    };
    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(())
}

#[derive(Debug)]
pub /*(crate)*/ struct Kernel(pub cl_kernel);

impl Drop for Kernel {
    fn drop(&mut self) {
        release_kernel(self).unwrap()
    }
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

pub(crate) fn create_kernel(program: &Program, str: &str) -> Result<Kernel, Error> {
    let mut err = 0;
    let cstring = CString::new(str).unwrap();
    let kernel = unsafe { clCreateKernel(program.0, cstring.as_ptr(), &mut err) };
    if err != 0 {
        return Err(Error::from(OCLErrorKind::from_value(err)));
    }
    Ok(Kernel(kernel))
}
pub(crate) fn create_kernels_in_program(program: &Program) -> Result<Vec<Rc<Kernel>>, Error> {
    let mut n_kernels: u32 = 0;
    let value =
        unsafe { clCreateKernelsInProgram(program.0, 0, std::ptr::null_mut(), &mut n_kernels) };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }

    let mut vec: Vec<usize> = vec![0; n_kernels as usize];
    let (ptr, len, cap) = (vec.as_mut_ptr(), vec.len(), vec.capacity());

    let mut kernels: Vec<Kernel> = unsafe {
        core::mem::forget(vec);
        Vec::from_raw_parts(ptr as *mut Kernel, len, cap)
    };
    let value = unsafe {
        clCreateKernelsInProgram(
            program.0,
            n_kernels,
            kernels.as_mut_ptr() as *mut cl_kernel,
            std::ptr::null_mut(),
        )
    };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }

    let kernels = kernels.into_iter().map(Rc::new).collect();

    Ok(kernels)
}

pub(crate) fn release_kernel(kernel: &mut Kernel) -> Result<(), Error> {
    let value = unsafe { clReleaseKernel(kernel.0) };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(())
}

pub fn set_kernel_arg(
    kernel: &Kernel,
    index: usize,
    arg: *const c_void,
    arg_size: usize,
    is_num: bool,
) -> Result<(), Error> {
    let ptr = if is_num {
        arg
    } else {
        &arg as *const *const c_void as *const c_void
    };

    let value = unsafe { clSetKernelArg(kernel.0, index as u32, arg_size, ptr) };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    Ok(())
}

pub fn enqueue_nd_range_kernel(
    cq: &CommandQueue,
    kernel: &Kernel,
    wd: usize,
    gws: &[usize; 3],
    lws: Option<&[usize; 3]>,
    offset: Option<[usize; 3]>,
) -> Result<(), Error> {
    let mut events = vec![std::ptr::null_mut(); 1];
    let lws = match lws {
        Some(lws) => lws.as_ptr(),
        None => std::ptr::null(),
    };
    let offset = match offset {
        Some(offset) => offset.as_ptr(),
        None => std::ptr::null(),
    };

    let value = unsafe {
        clEnqueueNDRangeKernel(
            cq.0,
            kernel.0,
            wd as u32,
            offset,
            gws.as_ptr(),
            lws,
            0,
            std::ptr::null(),
            events.as_mut_ptr() as *mut cl_event,
        )
    };
    if value != 0 {
        return Err(Error::from(OCLErrorKind::from_value(value)));
    }
    let e = Event(events[0]);
    wait_for_event(e)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use crate::{Buffer, CLDevice, OpenCL};

    #[test]
    fn test_multiplie_queues() -> crate::Result<()> {
        let device = CLDevice::new(0)?;
        let cl = OpenCL {
            kernel_cache: Default::default(),
            cache: Default::default(),
            inner: RefCell::new(device),
            graph: Default::default(),
            cpu: Default::default(),
        };

        let buf = Buffer::from((&cl, &[1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(buf.read(), vec![1, 2, 3, 4, 5, 6, 7]);

        let device = CLDevice::new(0)?;

        let cl1 = OpenCL {
            kernel_cache: Default::default(),
            cache: Default::default(),
            inner: RefCell::new(device),
            graph: Default::default(),
            cpu: Default::default(),
        };

        let buf = Buffer::from((&cl1, &[2, 2, 4, 4, 2, 1, 3]));
        assert_eq!(buf.read(), vec![2, 2, 4, 4, 2, 1, 3]);

        Ok(())
    }
}

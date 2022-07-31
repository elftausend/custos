use super::{
    api::{
        create_context, create_stream, cuInit, cuStreamDestroy,
        cu_read, cu_write,
        cublas::{create_handle, cublasDestroy_v2, cublasSetStream_v2, CublasHandle},
        cumalloc, device, Context, CudaIntDevice, Module, Stream,
    },
    cu_clear, CudaCache,
};
use crate::{
    deallocate_cache, get_device_count, AsDev, BaseDevice, CDatatype, CacheBuf,
    ClearBuf, Device, GenericBlas, VecRead, WriteBuf, Alloc, DeviceType, Buffer,
};
use std::{ptr::null_mut, cell::RefCell};

/// Used to perform calculations with a CUDA capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [CudaDevice].
#[derive(Debug)]
pub struct CudaDevice {
    pub modules: RefCell<Vec<Module>>,
    device: CudaIntDevice,
    ctx: Context,
    stream: Stream,
    handle: CublasHandle,
}

impl CudaDevice {
    pub fn new(idx: usize) -> crate::Result<CudaDevice> {
        unsafe {
            *get_device_count() += 1;
        }

        unsafe { cuInit(0) }.to_result()?;
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        let stream = create_stream()?;
        let handle = create_handle()?;
        unsafe { cublasSetStream_v2(handle.0, stream.0) }.to_result()?;

        Ok(CudaDevice { 
            modules: RefCell::new(vec![]),
            device,
            ctx,
            stream,
            handle,
        })
    }

    pub fn device(&self) -> &CudaIntDevice {
        &self.device
    }

    pub fn ctx(&self) -> &Context {
        &self.ctx
    }

    pub fn handle(&self) -> &CublasHandle {
        &self.handle
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        unsafe {
            let count = get_device_count();
            *count -= 1;
            deallocate_cache(*count);

            cublasDestroy_v2(self.handle.0);
            cuStreamDestroy(self.stream.0);
        }
    }
}

impl<T> Alloc<T> for CudaDevice {
    fn alloc(&self, len: usize) -> (*mut T, *mut std::ffi::c_void, u64) {
        let ptr = cumalloc::<T>(len).unwrap();
        // TODO: use unified mem if available -> i can't test this
        (null_mut(), null_mut(), ptr)
    }

    fn with_data(&self, data: &[T]) -> (*mut T, *mut std::ffi::c_void, u64) {
        let ptr = cumalloc::<T>(data.len()).unwrap();
        cu_write(ptr, data).unwrap();
        (null_mut(), null_mut(), ptr)
    }

    fn as_dev(&self) -> Device {
        Device {
            device_type: DeviceType::CUDA,
            device: self as *const CudaDevice as *mut u8
        }
    }
}

impl<T: Default + Copy> VecRead<T> for CudaDevice {
    fn read(&self, buf: &Buffer<T>) -> Vec<T> {
        assert!(
            buf.ptr.2 != 0,
            "called VecRead::read(..) on a non CUDA buffer"
        );
        let mut read = vec![T::default(); buf.len];
        cu_read(&mut read, buf.ptr.2).unwrap();
        read
    }
}

impl<T: CDatatype> ClearBuf<T> for CudaDevice {
    fn clear(&self, buf: &mut Buffer<T>) {
        cu_clear(self, buf).unwrap()
    }
}

impl<T> WriteBuf<T> for CudaDevice {
    fn write(&self, buf: &mut Buffer<T>, data: &[T]) {
        cu_write(buf.cu_ptr(), data).unwrap();
    }
}

impl<'a, T> CacheBuf<'a, T> for CudaDevice {
    fn cached(&self, len: usize) -> Buffer<T> {
        CudaCache::get::<T>(self, len)
    }
}

pub fn cu_cached<'a, T: Copy+Default>(device: &'a CudaDevice, len: usize) -> Buffer<'a, T> {
    device.cached(len)
}

impl AsDev for CudaDevice {}

impl<T: CDatatype + GenericBlas> BaseDevice<T> for CudaDevice {}
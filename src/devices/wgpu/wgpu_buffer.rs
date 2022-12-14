use core::{marker::PhantomData, mem::size_of};

use wgpu::util::DeviceExt;

use crate::Dealloc;

pub struct WGPUBuffer<T> {
    buf: wgpu::Buffer,
    _p: PhantomData<T>,
}

impl<T> WGPUBuffer<T> {
    pub fn new(device: &wgpu::Device, size: u64) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buf,
            _p: PhantomData,
        }
    }

    pub fn with_slice(device: &wgpu::Device, slice: &[T]) -> Self {
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: slice_u8_cast(slice),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        Self {
            buf,
            _p: PhantomData,
        }
    }
}

pub fn slice_u8_cast<T>(input: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * size_of::<T>()) }
}

impl<T> Dealloc<T> for WGPUBuffer<T> {
    unsafe fn dealloc(&mut self, len: usize) {
        // ...
    }
}

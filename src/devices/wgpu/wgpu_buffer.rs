use core::{marker::PhantomData, mem::size_of};
use wgpu::util::DeviceExt;

pub struct WGPUBuffer<T> {
    pub buf: wgpu::Buffer,
    pub _p: PhantomData<T>,
}

impl<T> WGPUBuffer<T> {
    pub fn new(device: &wgpu::Device, size: u64) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size * size_of::<T>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buf,
            _p: PhantomData,
        }
    }

    pub fn with_slice(device: &wgpu::Device, slice: &[T]) -> Self {
        let buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: slice_u8_cast(slice),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::MAP_READ,
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

pub fn slice_gen_cast<T>(input: &[u8]) -> &[T] {
    unsafe { std::slice::from_raw_parts(input.as_ptr() as *const T, input.len() / size_of::<T>()) }
}

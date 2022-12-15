use core::cell::RefCell;

use super::{shader_cache::ShaderCache, wgpu_buffer::*};

use crate::{Alloc, Device, Graph, GraphReturn, RawConv, CacheReturn, Cache, Node, Dealloc, Read};
use wgpu::{Adapter, Backends, Queue};

pub struct WGPU {
    pub adapter: Adapter,
    pub device: wgpu::Device,
    pub queue: Queue,
    pub graph: RefCell<Graph>,
    pub shader_cache: RefCell<ShaderCache>,
    pub cache: RefCell<Cache<WGPU>>
}

impl WGPU {
    pub fn new(backends: Backends) -> Option<WGPU> {
        let instance = wgpu::Instance::new(backends);

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .unwrap();

        Some(WGPU {
            adapter,
            device,
            queue,
            graph: Default::default(),
            shader_cache: Default::default(),
            cache: Default::default()
        })
    }
}

impl GraphReturn for WGPU {
    fn graph(&self) -> core::cell::RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

impl Device for WGPU {
    type Ptr<U, const N: usize> = WGPUBufPtr<U>;
    type Cache<const N: usize> = Cache<WGPU>;

    fn new() -> crate::Result<Self> {
        unimplemented!()
    }
}

impl<'a, T> Alloc<'a, T> for WGPU {
    fn alloc(&'a self, len: usize) -> WGPUBufPtr<T> {
        let wgpu_buf = WGPUBuffer::new(&self.device, len as u64);
        WGPUBufPtr { ptr: Box::leak(Box::new(wgpu_buf)) }
    }

    fn with_slice(&'a self, data: &[T]) -> WGPUBufPtr<T>
    where
        T: Clone,
    {
        let wgpu_buf = WGPUBuffer::with_slice(&self.device, data);
        WGPUBufPtr { ptr: Box::into_raw(Box::new(wgpu_buf)) }
    }
}

pub struct WGPUBufPtr<T> {
    pub ptr: *mut WGPUBuffer<T>
}

impl<T> WGPUBufPtr<T> {
    pub unsafe fn buf(&self) -> &wgpu::Buffer {
        &*(*self.ptr).buf
    }
}

impl<T> Dealloc<T> for WGPUBufPtr<T> {
    unsafe fn dealloc(&mut self, _len: usize) {
        drop(Box::from_raw(self.ptr));
    }
}

pub struct RawWGPUBuffer {
    pub ptr: *const u8,
    pub buffer: *mut wgpu::Buffer,
    node: Node
}

impl Drop for RawWGPUBuffer {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.buffer))
        }
    }
}

impl CacheReturn for WGPU {
    type CT = RawWGPUBuffer;

    fn cache(&self) -> core::cell::RefMut<crate::Cache<Self>>
    where
        Self: RawConv 
    {
        self.cache.borrow_mut()
    }
}

impl RawConv for WGPU {
    fn construct<T, const N: usize>(ptr: &Self::Ptr<T, N>, _len: usize, node: crate::Node) -> RawWGPUBuffer {
        unsafe {
            RawWGPUBuffer { 
                ptr: ptr.ptr as *const u8, 
                buffer: &mut *(*ptr.ptr).buf,
                node 
            }
        }
    }

    fn destruct<T, const N: usize>(ct: &RawWGPUBuffer) -> (Self::Ptr<T, N>, crate::Node) {
        (WGPUBufPtr {
            ptr: ct.ptr as *mut WGPUBuffer<T>,
        }
           , ct.node)
    }
}

impl<T: Default + Clone> Read<T, Self> for WGPU {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        Self: 'a;

    fn read<'a>(&self, buf: &'a crate::Buffer<T, Self>) -> Self::Read<'a> {
        self.read_to_vec(buf)
    }

    fn read_to_vec(&self, buf: &crate::Buffer<T, Self>) -> Vec<T>
    where
        T: Default + Clone 
    {

        self.queue.submit(None);

        let buf = unsafe {
            buf.ptr.buf()
        };

        let buf_slice = buf.slice(..);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        let Some(Ok(())) = pollster::block_on(receiver.receive()) else {
            panic!("Failed to read")
        };

        let data = buf_slice.get_mapped_range();
        slice_gen_cast::<T>(&data).to_vec()
    }
}
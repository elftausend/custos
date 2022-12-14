use core::cell::RefCell;

use super::wgpu_buffer::*;

use crate::{Alloc, Device, Graph, GraphReturn};
use wgpu::{Adapter, Backends, Queue};

pub struct WGPU {
    adapter: Adapter,
    device: wgpu::Device,
    queue: Queue,
    graph: RefCell<Graph>
}

impl WGPU {
    pub fn new(backends: Backends) -> Option<WGPU> {
        let instance = wgpu::Instance::new(backends);

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .unwrap();

        Some(WGPU {
            adapter,
            device,
            queue,
            graph: Default::default()
        })
    }
}

impl GraphReturn for WGPU {
    fn graph(&self) -> core::cell::RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

impl Device for WGPU {
    type Ptr<U, const N: usize> = WGPUBuffer<U>;
    type Cache<const N: usize> = ();

    fn new() -> crate::Result<Self> {
        unimplemented!()
    }
}

impl<'a, T> Alloc<'a, T> for WGPU {
    fn alloc(&'a self, len: usize) -> WGPUBuffer<T> {
        WGPUBuffer::new(&self.device, len as u64)
    }

    fn with_slice(&'a self, data: &[T]) -> WGPUBuffer<T>
    where
        T: Clone,
    {
        WGPUBuffer::with_slice(&self.device, data)
    }
}

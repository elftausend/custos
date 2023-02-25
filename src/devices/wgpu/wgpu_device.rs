use core::{cell::RefCell, fmt::Debug};

use super::{shader_cache::ShaderCache, wgpu_buffer::*, wgpu_clear};

use crate::{
    flag::AllocFlag, Alloc, Cache, CacheReturn, ClearBuf, Device, DeviceError, Graph, GraphReturn,
    Node, PtrType, RawConv, Read, Shape,
};
use wgpu::{Adapter, Backends, Queue};

pub struct WGPU {
    pub adapter: Adapter,
    pub device: wgpu::Device,
    pub queue: Queue,
    pub graph: RefCell<Graph>,
    pub shader_cache: RefCell<ShaderCache>,
    pub cache: RefCell<Cache<WGPU>>,
}

impl WGPU {
    pub fn new(backends: Backends) -> crate::Result<WGPU> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            dx12_shader_compiler: Default::default(),
        });

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok_or(DeviceError::WGPUDeviceReturn)?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))?;

        Ok(WGPU {
            adapter,
            device,
            queue,
            graph: Default::default(),
            shader_cache: Default::default(),
            cache: Default::default(),
        })
    }
}

impl GraphReturn for WGPU {
    fn graph(&self) -> core::cell::RefMut<Graph> {
        self.graph.borrow_mut()
    }
}

impl Device for WGPU {
    type Ptr<U, S: Shape> = WGPUBufPtr<U>;
    type Cache = Cache<WGPU>;

    fn new() -> crate::Result<Self> {
        unimplemented!()
    }
}

impl<T, S: Shape> Alloc<'_, T, S> for WGPU {
    fn alloc(&self, len: usize, flag: AllocFlag) -> WGPUBufPtr<T> {
        let wgpu_buf = WGPUBuffer::new(&self.device, len as u64);
        WGPUBufPtr {
            ptr: Box::leak(Box::new(wgpu_buf)),
            len,
            flag,
        }
    }

    fn with_slice(&self, data: &[T]) -> WGPUBufPtr<T>
    where
        T: Clone,
    {
        let wgpu_buf = WGPUBuffer::with_slice(&self.device, data);
        WGPUBufPtr {
            ptr: Box::into_raw(Box::new(wgpu_buf)),
            len: data.len(),
            flag: AllocFlag::None,
        }
    }
}

pub struct WGPUBufPtr<T> {
    pub ptr: *mut WGPUBuffer<T>,
    pub len: usize,
    pub flag: AllocFlag,
}

impl<T> WGPUBufPtr<T> {
    pub unsafe fn buf(&self) -> &wgpu::Buffer {
        &*(*self.ptr).buf
    }
}

impl<T> PtrType for WGPUBufPtr<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.flag
    }
}

impl<T> Drop for WGPUBufPtr<T> {
    fn drop(&mut self) {
        if !matches!(self.flag, AllocFlag::None | AllocFlag::BorrowedCache) {
            return;
        }

        unsafe { drop(Box::from_raw(self.ptr)) }
    }
}

pub struct RawWGPUBuffer {
    pub ptr: *const u8,
    pub buffer: *mut wgpu::Buffer,
    len: usize,
    node: Node,
}

impl Drop for RawWGPUBuffer {
    fn drop(&mut self) {
        unsafe { drop(Box::from_raw(self.buffer)) }
    }
}

impl CacheReturn for WGPU {
    type CT = RawWGPUBuffer;

    fn cache(&self) -> core::cell::RefMut<crate::Cache<Self>>
    where
        Self: RawConv,
    {
        self.cache.borrow_mut()
    }
}

impl RawConv for WGPU {
    fn construct<T, S: Shape>(
        ptr: &Self::Ptr<T, S>,
        len: usize,
        node: crate::Node,
    ) -> RawWGPUBuffer {
        unsafe {
            RawWGPUBuffer {
                ptr: ptr.ptr as *const u8,
                buffer: &mut *(*ptr.ptr).buf,
                len,
                node,
            }
        }
    }

    fn destruct<T, S: Shape>(
        ct: &RawWGPUBuffer,
        flag: AllocFlag,
    ) -> (Self::Ptr<T, S>, crate::Node) {
        (
            WGPUBufPtr {
                ptr: ct.ptr as *mut WGPUBuffer<T>,
                len: ct.len,
                flag,
            },
            ct.node,
        )
    }
}

impl<T: Default + Debug, S: Shape> ClearBuf<T, Self, S> for WGPU {
    /// Sets all the elements of a `WGPU` `Buffer` to zero / default.
    /// # Example
    /// ```
    /// use custos::{WGPU, Buffer, ClearBuf};
    /// fn main() -> custos::Result<()> {
    ///     let device = WGPU::new(wgpu::Backends::all())?;
    ///     let mut buf = Buffer::from((&device, [1, 5, 3, 4, 2]));
    ///     device.clear(&mut buf);
    ///
    ///     assert_eq!(buf.read(), [0, 0, 0, 0, 0]);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    fn clear(&self, buf: &mut crate::Buffer<T, Self, S>) {
        wgpu_clear(self, buf)
    }
}

impl<T: Default + Clone> Read<T, Self> for WGPU {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        Self: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a crate::Buffer<T, Self>) -> Self::Read<'a> {
        self.read_to_vec(buf)
    }

    fn read_to_vec(&self, buf: &crate::Buffer<T, Self>) -> Vec<T>
    where
        T: Default + Clone,
    {
        self.queue.submit(None);

        let buf = unsafe { buf.ptr.buf() };

        let buf_slice = buf.slice(..);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        let Some(Ok(())) = pollster::block_on(receiver.receive()) else {
            panic!("Failed to read")
        };

        let data = buf_slice.get_mapped_range();
        let read = slice_gen_cast::<T>(&data).to_vec();
        drop(data);
        buf.unmap();
        read
    }
}

use core::{cell::RefCell, fmt::Debug, ptr::null_mut};

use super::{
    launch_shader, shader_cache::ShaderCache, wgpu_buffer::*, wgpu_clear, AsBindingResource,
};

use crate::{
    flag::AllocFlag, Addons, AddonsReturn, Alloc, Cache, ClearBuf, Device, DeviceError, PtrConv,
    PtrType, Read, Shape,
};
use wgpu::{Adapter, Backends, Queue};

/// Used to perform calculations with an WGPU capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [WGPU].
/// # Example
/// ```
/// use custos::{WGPU, Read, Buffer, Error};
///
/// fn main() -> custos::Result<()> {
///     let device = WGPU::new(wgpu::Backends::all())?;
///     
///     let a = Buffer::from((&device, [1.3; 25]));
///     let out = device.read(&a);
///     
///     assert_eq!(out, vec![1.3; 5*5]);
///     Ok(())
/// }
/// ```
pub struct WGPU {
    /// The WGPU adapter
    pub adapter: Adapter,
    /// The WGPU device
    pub device: wgpu::Device,
    /// The WGPU queue
    pub queue: Queue,
    /// Caches compiled shaders for reuse
    pub shader_cache: RefCell<ShaderCache>,
    /// Provides additional functionality for the WGPU device. e.g. a cache, a gradient [`Tape`](crate::Tape), an optimizeable [`Graph`](crate::Graph) and a [`Cache`](crate::Cache).
    pub addons: Addons<WGPU>,
}

impl WGPU {
    /// Returns an [WGPU] device with the specified backends.
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
                features: wgpu::Features::default(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))?;

        Ok(WGPU {
            adapter,
            device,
            queue,
            shader_cache: Default::default(),
            addons: Default::default(),
        })
    }

    /// Launches a shader with the specified 'global work size' (`dispatch workgroups`) and arguments.
    #[inline]
    pub fn launch_kernel(&self, src: &str, gws: [u32; 3], args: &[impl AsBindingResource]) {
        launch_shader(self, src, gws, args)
    }
}

impl Default for WGPU {
    #[inline]
    fn default() -> Self {
        Self::new(Backends::PRIMARY).unwrap()
    }
}

impl Device for WGPU {
    type Ptr<U, S: Shape> = WGPUBufPtr<U>;
    type Cache = Cache<WGPU>;

    #[inline]
    fn new() -> crate::Result<Self> {
        Ok(WGPU::default())
    }
}

impl AddonsReturn for WGPU {
    #[inline]
    fn addons(&self) -> &Addons<Self> {
        &self.addons
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

/// A pointer to a WGPU buffer.
pub struct WGPUBufPtr<T> {
    /// The pointer to the buffer
    pub ptr: *mut WGPUBuffer<T>,
    /// The number of elements in the buffer
    pub len: usize,
    /// The allocation flag of the buffer
    pub flag: AllocFlag,
}

impl<T> WGPUBufPtr<T> {
    /// Returns a reference to the WGPU buffer.
    /// # Safety
    /// The buffer must not be dropped.
    #[inline]
    pub unsafe fn buf(&self) -> &wgpu::Buffer {
        &(*self.ptr).buf
    }
}

impl Default for WGPUBufPtr<u8> {
    #[inline]
    fn default() -> Self {
        WGPUBufPtr {
            ptr: null_mut(),
            len: 0,
            flag: AllocFlag::Wrapper,
        }
    }
}

impl<T> PtrType for WGPUBufPtr<T> {
    #[inline]
    fn size(&self) -> usize {
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

impl PtrConv for WGPU {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Ptr<T, IS>,
        flag: AllocFlag,
    ) -> Self::Ptr<Conv, OS> {
        WGPUBufPtr {
            ptr: ptr.ptr.cast(),
            len: ptr.len,
            flag,
        }
    }
}

impl<T: Default + Debug, S: Shape> ClearBuf<T, S> for WGPU {
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

impl<T: Default + Clone> Read<T> for WGPU {
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

use min_cl::CLDevice;

use min_cl::api::{create_buffer, enqueue_full_copy_buffer, MemFlags};

use super::{enqueue_kernel, AsClCvoidPtr, CLPtr};
use crate::{flag::AllocFlag, opencl::KernelLaunch};
use crate::{impl_device_traits, Shape, Unit};
use crate::{
    pass_down_use_gpu_or_cpu, Alloc, Base, Buffer, Cached, CachedCPU, CloneBuf, Device,
    IsShapeIndep, Module, OnDropBuffer, OnNewBuffer, Setup, WrappedData, CPU,
};

use core::ops::{Deref, DerefMut};
use std::fmt::Debug;

use min_cl::api::unified_ptr;

#[cfg(feature = "fork")]
use crate::ForkSetup;

/// Used to perform calculations with an OpenCL capable device.
/// To make new calculations invocable, a trait providing new operations should be implemented for [OpenCL].
/// # Example
/// ```
/// use custos::{OpenCL, Read, Buffer, Error, Base};
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::<Base>::new(0)?;
///     
///     let a = Buffer::from((&device, [1.3; 25]));
///     let out = a.read();
///     
///     assert_eq!(out, vec![1.3; 5*5]);
///     Ok(())
/// }
/// ```
pub struct OpenCL<Mods = Base> {
    pub modules: Mods,
    /// The underlying OpenCL device.
    pub device: CLDevice,
    /// A [`CPU`] used for unified memory device switching.
    pub cpu: CachedCPU,
}

/// Short form for `OpenCL`
pub type CL<Mods> = OpenCL<Mods>;

impl_device_traits!(OpenCL);

impl<Mods> Deref for OpenCL<Mods> {
    type Target = CLDevice;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl<Mods> DerefMut for OpenCL<Mods> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.device
    }
}

impl<SimpleMods> OpenCL<SimpleMods> {
    /// Returns an [OpenCL] device at the specified device index.
    /// # Errors
    /// - No device was found at the given device index
    /// - some other OpenCL related errors
    #[inline]
    pub fn new<'a, NewMods>(device_idx: usize) -> crate::Result<OpenCL<NewMods>>
    where
        SimpleMods: Module<'a, OpenCL, Module = NewMods>,
        NewMods: Setup<OpenCL<NewMods>>,
    {
        OpenCL::<SimpleMods>::from_cl_device(CLDevice::new(device_idx)?)
    }

    pub fn from_cl_device<'a, NewMods>(device: CLDevice) -> crate::Result<OpenCL<NewMods>>
    where
        SimpleMods: Module<'a, OpenCL, Module = NewMods>,
        NewMods: Setup<OpenCL<NewMods>>,
    {
        let mut opencl = OpenCL {
            modules: SimpleMods::new(),
            cpu: CPU::<Cached<Base>>::new(),
            device,
        };
        opencl.unified_mem_check();
        NewMods::setup(&mut opencl)?;
        Ok(opencl)
    }

    /// Returns the fastest [OpenCL] device available in your system.
    #[inline]
    pub fn fastest<'a, NewMods>() -> crate::Result<OpenCL<NewMods>>
    where
        SimpleMods: Module<'a, OpenCL, Module = NewMods>,
        NewMods: Setup<OpenCL<NewMods>>,
    {
        OpenCL::<SimpleMods>::from_cl_device(CLDevice::fastest()?)
    }
}

impl OpenCL {
    #[inline]
    pub fn based(idx: usize) -> crate::Result<OpenCL<Base>> {
        OpenCL::<Base>::new(idx)
    }
}

impl<Mods> OpenCL<Mods> {
    pub fn unified_mem_check(&self) {
        #[cfg(unified_cl)]
        if !self.unified_mem() {
            panic!("
                Your selected compute device does not support unified memory! 
                You are probably using a laptop.
                Launch with environment variable `CUSTOS_USE_UNIFIED=false` or change `CUSTOS_CL_DEVICE_IDX=<idx:default=0>`.
                `CUSTOS_CL_DEVICE_IDX` is used to determine during compile-time if custos should be configured to use unified memory (reduces memory transfers).
            ")
        }
    }

    /// Sets the values of the attributes cache, kernel cache, graph and CPU to their default.
    /// This cleans up any accumulated allocations.
    pub fn reset(&'static mut self) {
        self.device.kernel_cache = Default::default();
        self.cpu = CPU::<Cached<Base>>::new();
    }

    /// Executes a cached OpenCL kernel.
    /// # Example
    ///
    /// ```
    /// use custos::{OpenCL, Buffer, Base};
    ///
    /// fn main() -> custos::Result<()> {
    ///     let device = OpenCL::<Base>::new(0)?;
    ///     let mut buf = Buffer::<f32, _>::new(&device, 10);
    ///
    ///     device.launch_kernel("
    ///      __kernel void add(__global float* buf, float num) {
    ///         int idx = get_global_id(0);
    ///         buf[idx] += num;
    ///      }
    ///     ", [buf.len(), 0, 0], None, &[&mut buf, &4f32])?;
    ///     
    ///     assert_eq!(buf.read_to_vec(), [4.0; 10]);    
    ///
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn launch_kernel(
        &self,
        src: &str,
        gws: [usize; 3],
        lws: Option<[usize; 3]>,
        args: &[&dyn AsClCvoidPtr],
    ) -> crate::Result<()> {
        self.device.launch_kernel(src, gws, lws, args)
    }
}

impl<Mods: OnDropBuffer> Device for OpenCL<Mods> {
    type Data<T: Unit, S: Shape> = Self::Wrap<T, Self::Base<T, S>>;
    type Base<U: Unit, S: Shape> = CLPtr<U>;
    type Error = ();

    fn new() -> Result<Self, Self::Error> {
        todo!()
        // OpenCL::<Base>::new(chosen_cl_idx())
    }
    #[inline(always)]
    fn base_to_data<T: Unit, S: Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn wrap_to_data<T: Unit, S: Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<T: Unit, S: Shape>(
        data: &Self::Data<T, S>,
    ) -> &Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<T: Unit, S: Shape>(
        data: &mut Self::Data<T, S>,
    ) -> &mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }
}

unsafe impl<Mods: OnDropBuffer> IsShapeIndep for OpenCL<Mods> {}

impl<Mods> Debug for OpenCL<Mods> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CLDevice {{
            name: {name:?},
            version: {version:?},
            max_mem_alloc_in_gb: {max_mem:?},
            unified_mem: {unified_mem},
        }}",
            name = self.name(),
            version = self.version(),
            unified_mem = self.unified_mem(),
            max_mem = self.max_mem_alloc_in_gb()
        )
    }
}

impl<Mods: OnDropBuffer, T: Unit> Alloc<T> for OpenCL<Mods> {
    fn alloc<S: Shape>(&self, mut len: usize, flag: AllocFlag) -> crate::Result<CLPtr<T>> {
        if S::LEN > len {
            len = S::LEN
        }

        let ptr =
            unsafe { create_buffer::<T>(self.ctx(), MemFlags::MemReadWrite as u64, len, None)? };

        let host_ptr = if self.unified_mem() {
            unsafe { self.device.unified_ptr(ptr, len) }?
        } else {
            std::ptr::null_mut()
        };

        Ok(CLPtr {
            ptr,
            host_ptr,
            len,
            flag,
        })
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<CLPtr<T>> {
        let ptr = unsafe {
            create_buffer::<T>(
                self.ctx(),
                MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
                data.len(),
                Some(data),
            )?
        };

        let host_ptr = if self.unified_mem() {
            unsafe { unified_ptr::<T>(self.queue(), ptr, data.len(), None)? }
        } else {
            std::ptr::null_mut()
        };

        Ok(CLPtr {
            ptr,
            host_ptr,
            len: data.len(),
            flag: AllocFlag::None,
        })
    }
}

impl<'a, T: Unit, Mods: OnDropBuffer + OnNewBuffer<T, Self, ()>> CloneBuf<'a, T> for OpenCL<Mods> {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, Self>) -> Buffer<'a, T, Self> {
        let cloned = Buffer::new(self, buf.len());
        let event = unsafe {
            enqueue_full_copy_buffer::<T>(
                self.queue(),
                buf.base().ptr,
                cloned.base().ptr,
                buf.len(),
                None,
            )
            .unwrap()
        };
        event.wait().unwrap();
        cloned
    }
}

#[cfg(feature = "fork")]
impl<Mods> ForkSetup for OpenCL<Mods> {
    #[inline]
    fn fork_setup(&mut self) {}

    #[inline]
    fn has_unified_mem(&self) -> bool {
        self.unified_mem()
    }
}

pass_down_use_gpu_or_cpu!(OpenCL);

impl<Mods: crate::RunModule<Self>> crate::Run for OpenCL<Mods> {
    #[inline]
    unsafe fn run(&self) -> crate::Result<()> {
        self.modules.run(self)
    }
}

#[cfg(feature = "lazy")]
impl<Mods> crate::LazySetup for OpenCL<Mods> {}

#[cfg(feature = "lazy")]
impl<Mods> crate::LazyRun for OpenCL<Mods> {}

#[cfg(test)]
mod tests {
    use min_cl::api::OCLErrorKind;

    use crate::{opencl::cl_device::CLDevice, Alloc, Base, Buffer, Cached, OpenCL, CPU};

    #[test]
    fn test_fastest_cl_device() {
        // let dev = min_cl::CLDevice::fastest().unwrap();
        // println!("dev: {}", dev.name().unwrap());
        let _device = OpenCL::<Base>::fastest().unwrap();
    }

    #[test]
    fn test_multiplie_queues() -> crate::Result<()> {
        let device = CLDevice::new(0)?;
        let cl = OpenCL {
            device,
            cpu: CPU::<Cached<Base>>::new(),
            modules: crate::Base,
        };

        let buf = Buffer::from((&cl, &[1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(buf.read(), vec![1, 2, 3, 4, 5, 6, 7]);

        let device = CLDevice::new(0)?;

        let cl1 = OpenCL {
            device,
            cpu: CPU::<Cached<Base>>::new(),
            modules: crate::Base,
        };

        let buf = Buffer::from((&cl1, &[2, 2, 4, 4, 2, 1, 3]));
        assert_eq!(buf.read(), vec![2, 2, 4, 4, 2, 1, 3]);

        Ok(())
    }

    #[test]
    fn test_size_zero_alloc_cl() {
        let device = OpenCL::based(0).unwrap();
        let res = Alloc::<i32>::alloc_from_slice::<()>(&device, &[]);
        if let Err(e) = res {
            let e = e.downcast_ref::<OCLErrorKind>().unwrap();
            if e != &OCLErrorKind::InvalidBufferSize {
                panic!()
            }
        } else {
            panic!()
        }

        let res = Alloc::<i32>::alloc::<()>(&device, 0, crate::flag::AllocFlag::None);
        if let Err(e) = res {
            let e = e.downcast_ref::<OCLErrorKind>().unwrap();
            if e != &OCLErrorKind::InvalidBufferSize {
                panic!()
            }
        } else {
            panic!()
        }

        let res = Alloc::<i32>::alloc_from_vec::<()>(&device, vec![]);
        if let Err(e) = res {
            let e = e.downcast_ref::<OCLErrorKind>().unwrap();
            if e != &OCLErrorKind::InvalidBufferSize {
                panic!()
            }
        } else {
            panic!()
        }
    }
}

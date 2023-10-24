use min_cl::CLDevice;

use min_cl::api::{create_buffer, enqueue_full_copy_buffer, MemFlags};

use super::{enqueue_kernel, AsClCvoidPtr, CLPtr};
use crate::flag::AllocFlag;
use crate::{
    impl_buffer_hook_traits, impl_retriever, pass_down_optimize_mem_graph,
    pass_down_use_gpu_or_cpu, Alloc, Base, Buffer, Cached, CachedCPU, CloneBuf, Device, Module,
    OnDropBuffer, Setup, CPU,
};
use crate::{PtrConv, Shape};

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
///     let out = device.read(&a);
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
    pub cpu: CachedCPU, // TODO: this cpu does not cache buffers, which is a problem for construct_buffer (add #[cfg(unified_cl)])
}

/// Short form for `OpenCL`
pub type CL = OpenCL;

/*impl<Mods> HasCPU<Base> for OpenCL<Mods> {
    #[inline]
    fn cpu(&self) -> &crate::CPU {
        &self.cpu
    }
}*/

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

impl_buffer_hook_traits!(OpenCL);
impl_retriever!(OpenCL);

impl<SimpleMods> OpenCL<SimpleMods> {
    /// Returns an [OpenCL] device at the specified device index.
    /// # Errors
    /// - No device was found at the given device index
    /// - some other OpenCL related errors
    #[inline]
    pub fn new<NewMods>(device_idx: usize) -> crate::Result<OpenCL<NewMods>>
    where
        SimpleMods: Module<OpenCL, Module = NewMods>,
        NewMods: Setup<OpenCL<NewMods>>,
    {
        let mut opencl = OpenCL {
            modules: SimpleMods::new(),
            device: CLDevice::new(device_idx)?,
            cpu: CPU::<Cached<Base>>::new(),
        };
        NewMods::setup(&mut opencl)?;
        Ok(opencl)
    }

    /// Returns the fastest [OpenCL] device available in your system.
    #[inline]
    pub fn fastest<NewMods>() -> crate::Result<OpenCL<NewMods>>
    where
        SimpleMods: Module<OpenCL, Module = NewMods>,
        NewMods: Setup<OpenCL<NewMods>>,
    {
        let mut opencl = OpenCL {
            modules: SimpleMods::new(),
            device: CLDevice::fastest()?,
            cpu: CPU::<Cached<Base>>::new(),
        };
        NewMods::setup(&mut opencl)?;
        Ok(opencl)
    }
}

impl<Mods> OpenCL<Mods> {
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
        enqueue_kernel(self, src, gws, lws, args)
    }
}

impl<Mods: OnDropBuffer> Device for OpenCL<Mods> {
    type Data<U, S: Shape> = CLPtr<U>;
    type Error = ();

    fn new() -> Result<Self, Self::Error> {
        todo!()
        // OpenCL::<Base>::new(chosen_cl_idx())
    }
}

impl<Mods: OnDropBuffer, OtherMods: OnDropBuffer> PtrConv<OpenCL<OtherMods>> for OpenCL<Mods> {
    #[inline]
    unsafe fn convert<T, IS, Conv, OS>(
        ptr: &Self::Data<T, IS>,
        flag: AllocFlag,
    ) -> Self::Data<Conv, OS>
    where
        IS: Shape,
        OS: Shape,
    {
        CLPtr {
            ptr: ptr.ptr,
            host_ptr: ptr.host_ptr.cast(),
            len: ptr.len,
            flag,
        }
    }
}

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

impl<Mods: OnDropBuffer, T> Alloc<T> for OpenCL<Mods> {
    fn alloc<S: Shape>(&self, mut len: usize, flag: AllocFlag) -> CLPtr<T> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        let ptr = create_buffer::<T>(self.ctx(), MemFlags::MemReadWrite as u64, len, None).unwrap();

        let host_ptr = if self.unified_mem() {
            unified_ptr::<T>(self.queue(), ptr, len).unwrap()
        } else {
            std::ptr::null_mut()
        };

        CLPtr {
            ptr,
            host_ptr,
            len,
            flag,
        }
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> CLPtr<T> {
        let ptr = create_buffer::<T>(
            self.ctx(),
            MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
            data.len(),
            Some(data),
        )
        .unwrap();

        let host_ptr = if self.unified_mem() {
            unified_ptr::<T>(self.queue(), ptr, data.len()).unwrap()
        } else {
            std::ptr::null_mut()
        };

        CLPtr {
            ptr,
            host_ptr,
            len: data.len(),
            flag: AllocFlag::None,
        }
    }
}

impl<'a, T> CloneBuf<'a, T> for OpenCL {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, OpenCL>) -> Buffer<'a, T, OpenCL> {
        let cloned = Buffer::new(self, buf.len());
        enqueue_full_copy_buffer::<T>(self.queue(), buf.data.ptr, cloned.data.ptr, buf.len())
            .unwrap();
        cloned
    }
}

#[cfg(feature = "fork")]
impl<Mods> ForkSetup for OpenCL<Mods> {
    #[inline]
    fn fork_setup(&mut self) {
        assert!(
            self.unified_mem(),
            "The selected device does not support unified memory."
        )
    }
}

pass_down_use_gpu_or_cpu!(OpenCL);
pass_down_optimize_mem_graph!(OpenCL);

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

#[cfg(feature = "autograd")]
impl<Mods: crate::TapeActions> crate::TapeActions for OpenCL<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<crate::Tape>> {
        self.modules.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<crate::Tape>> {
        self.modules.tape_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::{opencl::cl_device::CLDevice, Base, Buffer, Cached, OpenCL, CPU};

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
}

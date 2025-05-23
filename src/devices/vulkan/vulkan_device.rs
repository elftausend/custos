use ash::vk::{self, BufferUsageFlags};

use super::{AsVkShaderArgument, ShaderCache, VkArray, context::Context, launch_shader};
use crate::{
    Alloc, Base, Buffer, Device, DeviceError, IsShapeIndep, Module, Setup, Shape, Unit,
    WrappedData, impl_device_traits, pass_down_use_gpu_or_cpu,
    wgsl::{WgslDevice, WgslShaderLaunch, chosen_wgsl_idx},
};
use core::{
    cell::RefCell,
    ops::{Deref, DerefMut},
};
use std::rc::Rc;

pub struct VkDevice {
    pub context: Rc<Context>,
    pub shader_cache: RefCell<ShaderCache>,
}

impl VkDevice {
    pub fn new(idx: usize) -> crate::Result<Self> {
        let context = Rc::new(Context::new(idx)?);
        Ok(VkDevice {
            shader_cache: RefCell::new(ShaderCache::new(context.clone())),
            context,
        })
    }

    #[inline]
    pub fn launch_shader(
        &self,
        src: impl AsRef<str>,
        gws: [u32; 3],
        args: &[&dyn AsVkShaderArgument],
    ) -> crate::Result<()> {
        launch_shader(
            self.context.clone(),
            gws,
            &mut self.shader_cache.borrow_mut(),
            src,
            args,
        )
    }
}

impl<Mods> WgslShaderLaunch for Vulkan<Mods> {
    type ShaderArg = dyn AsVkShaderArgument;

    #[inline]
    fn launch_shader(
        &self,
        src: impl AsRef<str>,
        gws: [u32; 3],
        args: &[&dyn AsVkShaderArgument],
    ) -> crate::Result<()> {
        self.device.launch_shader(src, gws, args)
    }
}

pub struct Vulkan<Mods = Base> {
    pub modules: Mods,
    pub device: VkDevice,
}

impl<SimpleMods> Vulkan<SimpleMods> {
    #[inline]
    pub fn new<'a, NewMods>(idx: usize) -> crate::Result<Vulkan<NewMods>>
    where
        Self: 'a,
        SimpleMods: Module<'a, Vulkan, Module = NewMods>,
        NewMods: Setup<Vulkan<NewMods>>,
    {
        let mut vulkan = Vulkan {
            modules: SimpleMods::new(),
            device: VkDevice::new(idx)?,
        };
        NewMods::setup(&mut vulkan)?;
        Ok(vulkan)
    }
}

impl Default for Vulkan {
    fn default() -> Self {
        Self {
            modules: Default::default(),
            device: VkDevice::new(chosen_wgsl_idx()).expect("Could not create vulkan device."),
        }
    }
}

impl WgslDevice for Vulkan {
    #[inline]
    fn new(idx: usize) -> crate::Result<Self> {
        Vulkan::<Base>::new(idx)
    }
}

impl<Mods> Deref for Vulkan<Mods> {
    type Target = VkDevice;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl<Mods> DerefMut for Vulkan<Mods> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.device
    }
}

impl<Mods> Vulkan<Mods> {
    #[inline]
    pub fn context(&self) -> Rc<Context> {
        self.context.clone()
    }
}

impl_device_traits!(Vulkan);
pass_down_use_gpu_or_cpu!(Vulkan);

impl<Mods: WrappedData> Device for Vulkan<Mods> {
    type Data<'a, T: Unit, S: Shape> = Mods::Wrap<'a, T, Self::Base<T, S>>;
    type Base<T: Unit, S: Shape> = VkArray<T>;

    type Error = ();

    #[inline(always)]
    fn default_base_to_data<'a, T: Unit, S: Shape>(
        &'a self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn default_base_to_data_unbound<'a, T: Unit, S: Shape>(
        &self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.wrap_in_base_unbound(base)
    }

    #[inline(always)]
    fn wrap_to_data<'a, T: Unit, S: Shape>(
        &self,
        wrap: Self::Wrap<'a, T, Self::Base<T, S>>,
    ) -> Self::Data<'a, T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<'a, 'b, T: Unit, S: Shape>(
        data: &'b Self::Data<'a, T, S>,
    ) -> &'b Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<'a, 'b, T: Unit, S: Shape>(
        data: &'b mut Self::Data<'a, T, S>,
    ) -> &'b mut Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }
}

impl<Mods: WrappedData, T: Unit> Alloc<T> for Vulkan<Mods> {
    #[inline]
    fn alloc<S: Shape>(
        &self,
        len: usize,
        flag: crate::flag::AllocFlag,
    ) -> crate::Result<Self::Base<T, S>> {
        if len == 0 {
            return Err(DeviceError::ZeroLengthBuffer.into());
        }
        VkArray::new(
            self.context(),
            len,
            BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::TRANSFER_SRC
                | BufferUsageFlags::TRANSFER_DST,
            flag,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    #[inline]
    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        if data.is_empty() {
            return Err(DeviceError::ZeroLengthBuffer.into());
        }
        VkArray::from_slice(
            self.context(),
            data,
            BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::TRANSFER_SRC
                | BufferUsageFlags::TRANSFER_DST,
            crate::flag::AllocFlag::None,
        )
    }
}

#[cfg(feature = "fork")]
impl<Mods> crate::ForkSetup for Vulkan<Mods> {}

unsafe impl<Mods: WrappedData> IsShapeIndep for Vulkan<Mods> {}

#[cfg(test)]
mod tests {
    use crate::{Base, Device};

    use super::Vulkan;

    pub fn add_one<Mods>(device: &Vulkan<Mods>, buf: ash::vk::Buffer) {
        let src = "
            @group(0)
            @binding(0)
            var<storage, read_write> out: array<i32>;

            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) { 
                if global_id.x >= arrayLength(&out) {
                    return;    
                }
                out[global_id.x] += 1;
            }
        ";

        device.launch_shader(src, [1, 1, 1], &[&buf]).unwrap();
    }

    #[test]
    fn test_running_compute_shader_with_vulkan_device() {
        let device = Vulkan::<Base>::new(0).unwrap();

        let buf = device.buffer([1, 2, 3, 4, 5, 9, 2, 3, 4, 3, 2]);
        add_one(&device, buf.data.buf);
        assert_eq!(&*buf.read(), [2, 3, 4, 5, 6, 10, 3, 4, 5, 4, 3])
    }

    #[test]
    fn test_using_multiple_compute_shaders() {
        let device = Vulkan::<Base>::new(0).unwrap();

        let out = device.buffer([1, 2, 3, 4, 5, 9, 2, 3, 4, 3, 2]);
        add_one(&device, out.data.buf);
        assert_eq!(&*out.read(), [2, 3, 4, 5, 6, 10, 3, 4, 5, 4, 3]);

        let lhs = device.buffer([2; 11]);
        let rhs = device.buffer([3; 11]);

        let src = "@group(0)
            @binding(0)
            var<storage, read_write> a: array<i32>;
            
            @group(0)
            @binding(1)
            var<storage, read_write> b: array<i32>;
    
            @group(0)
            @binding(2)
            var<storage, read_write> out: array<i32>;
            
            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return; 
                }
                
                out[global_id.x] += a[global_id.x] + b[global_id.x];
            }
        ";

        device
            .launch_shader(
                src,
                [1, 1, 1],
                &[&lhs.data.buf, &rhs.data.buf, &out.data.buf],
            )
            .unwrap();
        assert_eq!(&*out.read(), [7, 8, 9, 10, 11, 15, 8, 9, 10, 9, 8])
    }

    #[cfg(feature = "autograd")]
    #[test]
    fn test_vulkan_autograd() {
        use crate::{Autograd, Cached};

        let dev = Vulkan::<Cached<Autograd<Base>>>::new(0).unwrap();

        let lhs = dev.buffer([1, 2, 3, 4]);
        lhs.grad();
    }
}

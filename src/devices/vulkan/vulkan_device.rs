use ash::vk::BufferUsageFlags;

use super::{context::Context, launch_shader, AsVkShaderArgument, ShaderCache, VkArray};
use crate::{
    flag::AllocFlag, impl_buffer_hook_traits, impl_retriever, pass_down_grad_fn,
    pass_down_optimize_mem_graph, pass_down_tape_actions, pass_down_use_gpu_or_cpu, Alloc, Base,
    Buffer, Device, Module, OnDropBuffer, PtrConv, Setup, Shape,
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
        Ok(VkDevice {
            context: Rc::new(Context::new(idx)?),
            shader_cache: Default::default(),
        })
    }

    #[inline]
    pub fn launch_shader(
        &self,
        gws: [u32; 3],
        src: impl AsRef<str>,
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

pub struct Vulkan<Mods = Base> {
    pub modules: Mods,
    pub device: VkDevice,
}

impl<SimpleMods> Vulkan<SimpleMods> {
    #[inline]
    pub fn new<NewMods>(idx: usize) -> crate::Result<Vulkan<NewMods>>
    where
        SimpleMods: Module<Vulkan, Module = NewMods>,
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

impl_retriever!(Vulkan);
impl_buffer_hook_traits!(Vulkan);
pass_down_use_gpu_or_cpu!(Vulkan);
#[cfg(feature = "graph")]
pass_down_optimize_mem_graph!(Vulkan);

impl<Mods: OnDropBuffer> Device for Vulkan<Mods> {
    type Data<T, S: Shape> = VkArray<T>;

    type Error = ();
}

impl<Mods: OnDropBuffer, T> Alloc<T> for Vulkan<Mods> {
    #[inline]
    fn alloc<S: Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Data<T, S> {
        VkArray::new(self.context(), len, BufferUsageFlags::STORAGE_BUFFER, flag)
            .expect("Could not create VkArray")
    }

    #[inline]
    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone,
    {
        VkArray::from_slice(
            self.context(),
            data,
            BufferUsageFlags::STORAGE_BUFFER,
            crate::flag::AllocFlag::None,
        )
        .expect("Could not create VkArray")
    }
}

#[cfg(feature = "fork")]
impl<Mods> crate::ForkSetup for Vulkan<Mods> {}

pass_down_tape_actions!(Vulkan);
pass_down_grad_fn!(Vulkan);

// impl for all devices
impl<Mods: OnDropBuffer, OtherMods: OnDropBuffer> PtrConv<Vulkan<OtherMods>> for Vulkan<Mods> {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &VkArray<T>,
        flag: AllocFlag,
    ) -> VkArray<Conv> {
        VkArray {
            len: data.len,
            buf: data.buf,
            mem: data.mem,
            context: data.context.clone(),
            mapped_ptr: data.mapped_ptr as *mut Conv,
            flag,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Base, Device, HostPtr};

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

        device.launch_shader([1, 1, 1], src, &[&buf]).unwrap();
    }

    #[test]
    fn test_running_compute_shader_with_vulkan_device() {
        let device = Vulkan::<Base>::new(0).unwrap();

        let buf = device.buffer([1, 2, 3, 4, 5, 9, 2, 3, 4, 3, 2]);
        add_one(&device, buf.data.buf);
        assert_eq!(buf.as_slice(), [2, 3, 4, 5, 6, 10, 3, 4, 5, 4, 3])
    }

    #[test]
    fn test_using_multiple_compute_shaders() {
        let device = Vulkan::<Base>::new(0).unwrap();

        let out = device.buffer([1, 2, 3, 4, 5, 9, 2, 3, 4, 3, 2]);
        add_one(&device, out.data.buf);
        assert_eq!(out.as_slice(), [2, 3, 4, 5, 6, 10, 3, 4, 5, 4, 3]);

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
                [1, 1, 1],
                src,
                &[&lhs.data.buf, &rhs.data.buf, &out.data.buf],
            )
            .unwrap();
        assert_eq!(out.as_slice(), [7, 8, 9, 10, 11, 15, 8, 9, 10, 9, 8])
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

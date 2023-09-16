use super::{context::Context, launch_shader, ShaderCache, VkArray};
use crate::{
    flag::AllocFlag, impl_buffer_hook_traits, impl_retriever, Alloc, Base, Buffer, Device,
    MainMemory, Module, OnDropBuffer, PtrConv, Setup, Shape,
};
use core::cell::RefCell;
use std::rc::Rc;

pub struct Vulkan<Mods = Base> {
    pub modules: Mods,
    pub context: Rc<Context>,
    pub shader_cache: RefCell<ShaderCache>,
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
            context: Rc::new(Context::new(idx)?),
            shader_cache: Default::default(),
        };
        NewMods::setup(&mut vulkan)?;
        Ok(vulkan)
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

impl<Mods: OnDropBuffer> Device for Vulkan<Mods> {
    type Data<T, S: Shape> = VkArray<T>;

    type Error = ();
}

impl<Mods: OnDropBuffer, T> Alloc<T> for Vulkan<Mods> {
    #[inline]
    fn alloc<S: Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Data<T, S> {
        VkArray::new(self.context(), len, flag).expect("Could not create VkArray")
    }

    #[inline]
    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone,
    {
        VkArray::from_slice(self.context(), data, crate::flag::AllocFlag::None)
            .expect("Could not create VkArray")
    }
}

impl<Mods: OnDropBuffer> MainMemory for Vulkan<Mods> {
    #[inline]
    fn as_ptr<T, S: Shape>(ptr: &Self::Data<T, S>) -> *const T {
        ptr.mapped_ptr
    }

    #[inline]
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Data<T, S>) -> *mut T {
        ptr.mapped_ptr
    }
}

#[cfg(feature = "autograd")]
impl<Mods: crate::TapeActions> crate::TapeActions for Vulkan<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<crate::Tape>> {
        self.modules.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<crate::Tape>> {
        self.modules.tape_mut()
    }
}

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

impl<Mods> Vulkan<Mods> {
    #[inline]
    pub fn launch_shader(
        &self,
        gws: [u32; 3],
        src: impl AsRef<str>,
        args: &[ash::vk::Buffer],
    ) -> crate::Result<()> {
        launch_shader(
            &self.context.device,
            gws,
            &mut self.shader_cache.borrow_mut(),
            self.context.command_buffer,
            self.context.compute_family_idx,
            src,
            args,
        )
    }
}

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

        device.launch_shader([1, 1, 1], src, &[buf]).unwrap();
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
            .launch_shader([1, 1, 1], src, &[lhs.data.buf, rhs.data.buf, out.data.buf])
            .unwrap();
        assert_eq!(out.as_slice(), [7, 8, 9, 10, 11, 15, 8, 9, 10, 9, 8])
    }
}

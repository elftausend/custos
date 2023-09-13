use super::{context::Context, VkArray};
use crate::{Shape, Buffer, Base, Module, Setup, Alloc, Device, OnDropBuffer, impl_retriever, impl_buffer_hook_traits, MainMemory};
use std::rc::Rc;

pub struct Vulkan<Mods = Base> {
    pub modules: Mods,
    pub context: Rc<Context>,
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
        };
        NewMods::setup(&mut vulkan);
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
        T: Clone 
    { 
        VkArray::from_slice(self.context(), data, crate::flag::AllocFlag::None).expect("Could not create VkArray")
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

#[cfg(test)]
mod tests {
    use crate::{Base, Buffer, Device};

    use super::Vulkan;

    #[test]
    fn test_running_compute_shader_with_vulkan_device() {
        let device = Vulkan::<Base>::new(0).unwrap();

        let buf = device.buffer([1, 2, 3]);
    }
}

use core::convert::Infallible;

use crate::{
    AddOperation, Alloc, Base, Buffer, Device, HasId, IsShapeIndep, Module, OnDropBuffer,
    OnNewBuffer, Parents, PtrType, Retrieve, Retriever, Setup, Shape, Unit, WrappedData,
};

use super::{WgslDevice, WgslShaderLaunch};

pub struct Wgsl<D: Device, Mods = Base> {
    pub modules: Mods,
    pub backend: D,
}

impl<SimpleMods, D: WgslDevice + Device + Default> Wgsl<D, SimpleMods> {
    #[inline]
    pub fn new<'a, NewMods>(idx: usize) -> crate::Result<Wgsl<D, NewMods>>
    where
        Self: 'a,
        D: 'a,
        SimpleMods: Module<'a, Wgsl<D>, Module = NewMods>,
        NewMods: Setup<Wgsl<D, NewMods>>,
    {
        let mut wgsl = Wgsl {
            modules: SimpleMods::new(),
            backend: WgslDevice::new(idx)?,
        };
        NewMods::setup(&mut wgsl)?;
        Ok(wgsl)
    }
}

impl<D: Device, Mods: OnDropBuffer> Device for Wgsl<D, Mods> {
    type Base<T: Unit, S: crate::Shape> = D::Base<T, S>;

    type Data<T: Unit, S: crate::Shape> = Mods::Wrap<T, Self::Base<T, S>>;

    type Error = Infallible;

    #[inline]
    fn base_to_data<T: Unit, S: crate::Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline]
    fn wrap_to_data<T: Unit, S: crate::Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline]
    fn data_as_wrap<T: Unit, S: crate::Shape>(
        data: &Self::Data<T, S>,
    ) -> &Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline]
    fn data_as_wrap_mut<T: Unit, S: crate::Shape>(
        data: &mut Self::Data<T, S>,
    ) -> &mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }
}

impl<D: Device, Mods: WrappedData> WrappedData for Wgsl<D, Mods> {
    type Wrap<T, Base: HasId + PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: crate::HasId + crate::PtrType>(
        &self,
        base: Base,
    ) -> Self::Wrap<T, Base> {
        self.modules.wrap_in_base(base)
    }

    #[inline]
    fn wrapped_as_base<T, Base: crate::HasId + crate::PtrType>(
        wrap: &Self::Wrap<T, Base>,
    ) -> &Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: crate::HasId + crate::PtrType>(
        wrap: &mut Self::Wrap<T, Base>,
    ) -> &mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}
impl<D: Device, Mods: OnDropBuffer> OnDropBuffer for Wgsl<D, Mods> {
    #[inline]
    fn on_drop_buffer<T: Unit, D1: Device, S: crate::Shape>(
        &self,
        device: &D1,
        buf: &crate::Buffer<T, D1, S>,
    ) {
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<'dev, D: Device, Mods: OnNewBuffer<'dev, T, D1, S>, T: Unit, D1: Device, S: Shape>
    OnNewBuffer<'dev, T, D1, S> for Wgsl<D, Mods>
{
    #[inline]
    unsafe fn on_new_buffer(&self, device: &'dev D1, new_buf: &crate::Buffer<'dev, T, D1, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}

unsafe impl<D: Device, Mods: OnDropBuffer> IsShapeIndep for Wgsl<D, Mods> {}

impl<T: Unit, D: Alloc<T>, Mods: OnDropBuffer> Alloc<T> for Wgsl<D, Mods> {
    #[inline]
    fn alloc<S: Shape>(
        &self,
        len: usize,
        flag: crate::flag::AllocFlag,
    ) -> crate::Result<Self::Base<T, S>> {
        self.backend.alloc(len, flag)
    }

    #[inline]
    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        self.backend.alloc_from_slice(data)
    }

    #[inline]
    fn alloc_from_vec<S: Shape>(&self, vec: Vec<T>) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        self.backend.alloc_from_slice(&vec)
    }

    #[inline]
    fn alloc_from_array<S: Shape>(&self, array: S::ARR<T>) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        let stack_array = crate::StackArray::<S, T>::from_array(array);
        self.backend.alloc_from_slice(stack_array.flatten())
    }
}

impl<D: Device + WgslShaderLaunch, Mods: OnDropBuffer> WgslShaderLaunch for Wgsl<D, Mods> {
    type ShaderArg = D::ShaderArg;

    #[inline]
    fn launch_shader(
        &self,
        src: impl AsRef<str>,
        gws: [u32; 3],
        args: &[&Self::ShaderArg],
    ) -> crate::Result<()> {
        self.backend.launch_shader(src, gws, args)
    }
}

impl<D: Device + Alloc<T>, T: Unit, Mods: Retrieve<Self, T, S>, S: Shape> Retriever<T, S>
    for Wgsl<D, Mods>
{
    #[inline]
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Buffer<T, Self, S>> {
        let data = unsafe { self.modules.retrieve::<NUM_PARENTS>(self, len, parents) }?;
        let buf = Buffer {
            data,
            device: Some(self),
        };
        self.modules.on_retrieve_finish(&buf);
        Ok(buf)
    }
}

impl<D: Device, Mods: AddOperation> AddOperation for Wgsl<D, Mods> {
    fn add_op<Args: Parents<N> + crate::AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) -> crate::Result<()> {
        self.modules.add_op(args, op)
    }

    #[inline]
    fn ops_count(&self) -> usize {
        self.modules.ops_count()
    }

    #[inline]
    fn set_lazy_enabled(&self, enabled: bool) {
        self.modules.set_lazy_enabled(enabled)
    }

    #[inline]
    fn is_lazy_enabled(&self) -> bool {
        self.modules.is_lazy_enabled()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "vulkan")]
    use crate::{Device, Vulkan};

    use super::Wgsl;

    #[cfg(feature = "vulkan")]
    #[test]
    fn test_wgsl_wrapper_vk() {
        use crate::Vulkan;

        let dev = Wgsl::<Vulkan>::new(0).unwrap();
        let _x = dev.buffer([1, 2, 3]);
    }

    #[cfg(feature = "webgl")]
    #[test]
    fn test_wgsl_wrapper_webgl() {
        use crate::{Device, WebGL};

        let dev = Wgsl::<WebGL>::new(0).unwrap();
        let _x = dev.buffer([1f32, 2., 3.]);
    }
}

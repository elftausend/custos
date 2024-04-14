use core::convert::Infallible;

use crate::{
    Base, Device, HasId, Module, OnDropBuffer, OnNewBuffer, PtrType, Setup, Shape, WrappedData
};

pub struct Wgsl<D: Device, Mods = Base> {
    pub modules: Mods,
    pub backend: D,
}

impl<SimpleMods, D: Device + Default> Wgsl<D, SimpleMods> {
    #[inline]
    pub fn new<NewMods>() -> crate::Result<Wgsl<D, NewMods>>
    where
        SimpleMods: Module<Wgsl<D>, Module = NewMods>,
        NewMods: Setup<Wgsl<D, NewMods>>,
    {
        let mut wgsl = Wgsl {
            modules: SimpleMods::new(),
            backend: D::default(),
        };
        NewMods::setup(&mut wgsl)?;
        Ok(wgsl)
    }
}

impl<D: Device, Mods: OnDropBuffer> Device for Wgsl<D, Mods> {
    type Base<T, S: crate::Shape> = D::Base<T, S>;

    type Data<T, S: crate::Shape> = Mods::Wrap<T, Self::Base<T, S>>;

    type Error = Infallible;

    #[inline]
    fn base_to_data<T, S: crate::Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline]
    fn wrap_to_data<T, S: crate::Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline]
    fn data_as_wrap<T, S: crate::Shape>(
        data: &Self::Data<T, S>,
    ) -> &Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline]
    fn data_as_wrap_mut<T, S: crate::Shape>(
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
    fn on_drop_buffer<T, D1: Device, S: crate::Shape>(&self, device: &D1, buf: &crate::Buffer<T, D1, S>) {
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<D: Device, Mods: OnNewBuffer<T, D1, S>, T, D1: Device, S: Shape> OnNewBuffer<T, D1, S> for Wgsl<D, Mods> {
    #[inline]
    fn on_new_buffer(&self, device: &D1, new_buf: &crate::Buffer<T, D1, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}


#[cfg(test)]
mod tests {
    use crate::{Base, Vulkan};

    use super::Wgsl;

    #[test]
    fn test_wgsl_wrapper() {
        let dev = Wgsl::<Vulkan>::new().unwrap();
    }
}

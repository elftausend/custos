use crate::{Base, Buffer, Device, Shape, Unit, WrappedData, impl_buffer_hook_traits};

pub struct CUDA<Mods = Base> {
    pub modules: Mods,
}

impl<Mods: WrappedData> Device for CUDA<Mods> {
    type Data<'a, U: Unit, S: Shape> = Mods::Wrap<'a, U, crate::Num<U>>;
    type Base<T: Unit, S: Shape> = crate::Num<T>;
    type Error = crate::DeviceError;

    fn new() -> core::result::Result<Self, Self::Error> {
        #[cfg(not(feature = "std"))]
        {
            unimplemented!("CUDA is not available. Enable the `cpu` feature to use the CPU.")
        }

        #[cfg(feature = "std")]
        Err(crate::DeviceError::CPUDeviceNotAvailable)
    }

    fn default_base_to_data<'a, T: Unit, S: Shape>(
        &'a self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.modules.wrap_in_base(base)
    }

    fn default_base_to_data_unbound<'a, T: Unit, S: Shape>(
        &self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.modules.wrap_in_base_unbound(base)
    }

    fn wrap_to_data<'a, T: Unit, S: Shape>(
        &self,
        wrap: Self::Wrap<'a, T, Self::Base<T, S>>,
    ) -> Self::Data<'a, T, S> {
        wrap
    }

    fn data_as_wrap<'a, 'b, T: Unit, S: Shape>(
        data: &'b Self::Data<'a, T, S>,
    ) -> &'b Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }

    fn data_as_wrap_mut<'a, 'b, T: Unit, S: Shape>(
        data: &'b mut Self::Data<'a, T, S>,
    ) -> &'b mut Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }
}

impl_buffer_hook_traits!(CUDA);
crate::impl_wrapped_data!(CUDA);

impl<Mods> super::AsDeviceType for CUDA<Mods> {
    const DEVICE_TYPE: super::DeviceType = super::DeviceType::CUDA;
}

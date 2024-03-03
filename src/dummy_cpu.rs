use crate::{impl_buffer_hook_traits, Base, Buffer, Device, OnDropBuffer, Shape};

/// A dummy CPU. This only exists to make the code compile when the `cpu` feature is disabled
/// because the CPU is the default type `D` for [`Buffer`]s.
// TODO: Can be replaced with the standard cpu (now)
pub struct CPU<Mods = Base> {
    modules: Mods,
}

impl<Mods: OnDropBuffer> Device for CPU<Mods> {
    type Data<U, S: Shape> = Mods::Wrap<U, crate::Num<U>>;
    type Base<T, S: Shape> = crate::Num<T>;
    type Error = crate::DeviceError;

    fn new() -> core::result::Result<Self, Self::Error> {
        #[cfg(not(feature = "std"))]
        {
            unimplemented!("CPU is not available. Enable the `cpu` feature to use the CPU.")
        }

        #[cfg(feature = "std")]
        Err(crate::DeviceError::CPUDeviceNotAvailable.into())
    }

    fn base_to_data<T, S: Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.modules.wrap_in_base(base)
    }

    fn wrap_to_data<T, S: Shape>(&self, wrap: Self::Wrap<T, Self::Base<T, S>>) -> Self::Data<T, S> {
        wrap
    }

    fn data_as_wrap<'a, T, S: Shape>(
        data: &'a Self::Data<T, S>,
    ) -> &'a Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    fn data_as_wrap_mut<'a, T, S: Shape>(
        data: &'a mut Self::Data<T, S>,
    ) -> &'a mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }
}

impl_buffer_hook_traits!(CPU);
crate::impl_wrapped_data!(CPU);

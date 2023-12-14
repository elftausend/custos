use crate::{HasId, PtrType};

pub trait WrappedData {
    type Wrap<T, Base: HasId + PtrType>: HasId + PtrType;

    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base>;
}

#[macro_export]
macro_rules! impl_wrapped_data {
    ($device:ident) => {
        impl<Mods: $crate::WrappedData> $crate::WrappedData for $device<Mods> {
            type Wrap<T, Base: $crate::HasId + $crate::PtrType> = Mods::Wrap<T, Base>;

            #[inline]
            fn wrap_in_base<T, Base: $crate::HasId + $crate::PtrType>(
                &self,
                base: Base,
            ) -> Self::Wrap<T, Base> {
                self.modules.wrap_in_base(base)
            }
        }
    };
}

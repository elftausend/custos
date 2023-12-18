use crate::{HasId, PtrType};

pub trait WrappedData {
    type Wrap<T, Base: HasId + PtrType>: HasId + PtrType;

    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base>;
    fn wrapped_as_base<'a, T, Base: HasId + PtrType>(wrap: &'a Self::Wrap<T, Base>) -> &'a Base;
    fn wrapped_as_base_mut<'a, T, Base: HasId + PtrType>(
        wrap: &'a mut Self::Wrap<T, Base>,
    ) -> &'a mut Base;
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

            #[inline]
            fn wrapped_as_base<'a, T, Base: $crate::HasId + $crate::PtrType>(
                wrap: &'a Self::Wrap<T, Base>,
            ) -> &'a Base {
                Mods::wrapped_as_base(wrap)
            }

            #[inline]
            fn wrapped_as_base_mut<'a, T, Base: $crate::HasId + $crate::PtrType>(
                wrap: &'a mut Self::Wrap<T, Base>,
            ) -> &'a mut Base {
                Mods::wrapped_as_base_mut(wrap)
            }
        }
    };
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "lazy")]
    #[cfg(feature = "cpu")]
    #[test]
    fn test_wrapped_as_base() {
        use crate::{Base, Device, Lazy, CPU};

        let device = CPU::<Lazy<Base>>::new();
        let buf = device.buffer([1, 2, 3, 4]);
        let base = buf.base();
        assert_eq!(base.as_slice(), [1, 2, 3, 4])
    }
}

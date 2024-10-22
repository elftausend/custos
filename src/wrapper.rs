use crate::{HasId, PtrType, Unit};

pub trait WrappedData {
    type Wrap<T: Unit, Base: HasId + PtrType>: HasId + PtrType;

    fn wrap_in_base<T: Unit, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base>;
    #[track_caller]
    fn wrapped_as_base<T: Unit, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base;
    #[track_caller]
    fn wrapped_as_base_mut<T: Unit, Base: HasId + PtrType>(wrap: &mut Self::Wrap<T, Base>) -> &mut Base;
}

#[macro_export]
macro_rules! impl_wrapped_data {
    ($device:ident) => {
        impl<Mods: $crate::WrappedData> $crate::WrappedData for $device<Mods> {
            type Wrap<T: Unit, Base: $crate::HasId + $crate::PtrType> = Mods::Wrap<T, Base>;

            #[inline]
            fn wrap_in_base<T: Unit, Base: $crate::HasId + $crate::PtrType>(
                &self,
                base: Base,
            ) -> Self::Wrap<T, Base> {
                self.modules.wrap_in_base(base)
            }

            #[inline]
            fn wrapped_as_base<T: Unit, Base: $crate::HasId + $crate::PtrType>(
                wrap: &Self::Wrap<T, Base>,
            ) -> &Base {
                Mods::wrapped_as_base(wrap)
            }

            #[inline]
            fn wrapped_as_base_mut<T: Unit, Base: $crate::HasId + $crate::PtrType>(
                wrap: &mut Self::Wrap<T, Base>,
            ) -> &mut Base {
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

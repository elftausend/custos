use crate::{HasId, PtrType, Unit};

pub trait WrappedData {
    type Wrap<'a, T: Unit, Base: HasId + PtrType>: HasId + PtrType + 'a; 

    fn wrap_in_base<'a, T: Unit, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<'a, T, Base>;
    #[track_caller]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: HasId + PtrType>(wrap: &'b Self::Wrap<'a, T, Base>) -> &'b Base;
    #[track_caller]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: HasId + PtrType>(wrap: &'b mut Self::Wrap<'a, T, Base>) -> &'b mut Base;
}

#[macro_export]
macro_rules! impl_wrapped_data {
    ($device:ident) => {
        impl<Mods: $crate::WrappedData> $crate::WrappedData for $device<Mods> {
            type Wrap<'a, T: Unit, Base: $crate::HasId + $crate::PtrType> = Mods::Wrap<'a, T, Base>;

            #[inline]
            fn wrap_in_base<'a, T: Unit, Base: $crate::HasId + $crate::PtrType>(
                &self,
                base: Base,
            ) -> Self::Wrap<'a, T, Base> {
                self.modules.wrap_in_base(base)
            }

            #[inline]
            fn wrapped_as_base<'a, 'b, T: Unit, Base: $crate::HasId + $crate::PtrType>(
                wrap: &'b Self::Wrap<'a, T, Base>,
            ) -> &'b Base {
                Mods::wrapped_as_base(wrap)
            }

            #[inline]
            fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: $crate::HasId + $crate::PtrType>(
                wrap: &'b mut Self::Wrap<'a, T, Base>,
            ) -> &'b mut Base {
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

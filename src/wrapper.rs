use crate::{HasId, IsBasePtr, PtrType, Unit};

pub trait WrappedData {
    type Wrap<'a, T: Unit, Base: IsBasePtr>: PtrType + HasId + 'a;

    fn wrap_in_base<'a, T: Unit, Base: IsBasePtr>(&'a self, base: Base) -> Self::Wrap<'a, T, Base>;
    fn wrap_in_base_unbound<'a, T: Unit, Base: IsBasePtr>(
        &self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base>;
    #[track_caller]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base;
    #[track_caller]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base;
}

#[macro_export]
macro_rules! impl_wrapped_data {
    ($device:ident) => {
        impl<Mods: $crate::WrappedData> $crate::WrappedData for $device<Mods> {
            type Wrap<'a, T: Unit, Base: 'static + $crate::IsBasePtr> = Mods::Wrap<'a, T, Base>;

            #[inline]
            fn wrap_in_base<'a, T: Unit, Base: $crate::IsBasePtr>(
                &'a self,
                base: Base,
            ) -> Self::Wrap<'a, T, Base> {
                self.modules.wrap_in_base(base)
            }

            #[inline]
            fn wrap_in_base_unbound<'a, T: Unit, Base: $crate::IsBasePtr>(
                &self,
                base: Base,
            ) -> Self::Wrap<'a, T, Base> {
                self.modules.wrap_in_base_unbound(base)
            }

            #[inline]
            fn wrapped_as_base<'a, 'b, T: Unit, Base: $crate::IsBasePtr>(
                wrap: &'b Self::Wrap<'a, T, Base>,
            ) -> &'b Base {
                Mods::wrapped_as_base(wrap)
            }

            #[inline]
            fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: $crate::IsBasePtr>(
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

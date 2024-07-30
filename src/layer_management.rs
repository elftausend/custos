pub trait AddLayer<NewMods, SD> {
    type Wrapped;
    fn wrap_layer(inner_mods: NewMods) -> Self::Wrapped;
}

pub trait RemoveLayer<Mods> {
    fn inner_mods(self) -> Mods;
}

#[macro_export]
macro_rules! impl_remove_layer {
    ($module:ident, $($generics:tt),*) => {
        impl<$($generics),*> $crate::RemoveLayer<Mods> for $module<$($generics),*> {
            #[inline]
            fn inner_mods(self) -> Mods {
                self.modules
            }
        }
    };
    ($module:ident) => {
        $crate::impl_remove_layer!($module, Mods);
    };
}

pub trait AddLayer<NewMods, SD> {
    type Wrapped;
    fn wrap_layer(inner_mods: NewMods) -> Self::Wrapped;
}

pub trait RemoveLayer<Mods> {
    fn inner_mods(self) -> Mods;
}

#[macro_export]
macro_rules! impl_remove_layer {
    ($module:ident) => {
        impl<Mods> $crate::RemoveLayer<Mods> for $module<Mods> {
            #[inline]
            fn inner_mods(self) -> Mods {
                self.modules
            }
        }
    };
}

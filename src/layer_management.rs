
pub trait AddLayer<NewMods, SD> {
    type Wrapped;
    fn wrap_layer(inner_mods: NewMods) -> Self::Wrapped;
}

pub trait RemoveLayer<Mods> {
    fn inner_mods(self) -> Mods;
}

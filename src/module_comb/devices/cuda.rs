use crate::module_comb::LazySetup;

pub trait IsCuda {}

pub struct CUDA<Mods> {
    modules: Mods,
}

impl<Mods> IsCuda for CUDA<Mods> {}

impl<Mods> LazySetup for CUDA<Mods> {
    #[inline]
    fn lazy_setup(&mut self) {
        // switch to stream record mode for graph
    }
}

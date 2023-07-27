use crate::module_comb::LazySetup;

pub struct CUDA<Mods> {
    modules: Mods,
}

impl<Mods> LazySetup for CUDA<Mods> {
    #[inline]
    fn lazy_setup(&mut self) {
        // switch to stream record mode for graph
    }
}

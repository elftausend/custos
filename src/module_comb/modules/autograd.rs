use crate::module_comb::{Alloc, Module, Retrieve};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Autograd<Mods> {
    modules: Mods,
}

impl<NewMods, Mods: Module<SD, NewMods>, /*Module<SD, Module = Mods>,*/ SD> Module<SD, NewMods> for Autograd<Mods> {
    type Module = Autograd<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {  
        Autograd {
            modules: Mods::new(),
        }
    }
}


impl<Mods: Retrieve<D>, D> Retrieve<D> for Autograd<Mods> {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc,
    {
        println!("autograd: pass down retrieve");
        self.modules.retrieve(device, len)
    }
}

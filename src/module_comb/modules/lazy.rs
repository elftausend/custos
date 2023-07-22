use core::marker::PhantomData;

use crate::module_comb::{Module, Retrieve, Alloc, Setup};

#[derive(Debug, Default)]
pub struct Lazy<Mods> {
    mods: Mods,
}

pub trait LazySetup {
    fn setup(&mut self) {}
}

impl<Mods: Module<D, Module = Mods>, D: LazySetup> Module<D> for Lazy<Mods> {
    type Module = Lazy<Mods>;

    #[inline]
    fn new() -> Self::Module {
        Lazy {
            mods: Mods::new(),
        }
    }
}

impl<D: LazySetup, Mods: Setup<D>> Setup<D> for Lazy<Mods> {
    #[inline]
    fn setup(device: &mut D) {
        println!("setup lazy");
        Mods::setup(device)
    }
}

impl<Mods, D> Retrieve<D> for Lazy<Mods> {
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: crate::module_comb::Alloc,
    {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::module_comb::{CPU, Base, Alloc, CUDA};

    use super::Lazy;

    #[test]
    fn test_lazy_device_use() {
        /*let device = CUDA::<Lazy<Base>>::new();
        let data = device.alloc::<f32, ()>(10, crate::flag::AllocFlag::None);*/
    }
}

use custos::{Base, Module, Setup, CPU};

pub struct CustomModule<Mods> {
    pub mods: Mods,
}

impl<D, Mods: Module<D>> Module<D> for CustomModule<Mods> {
    type Module = CustomModule<Mods::Module>;

    fn new() -> Self::Module {
        CustomModule {
            mods: Mods::new(),
        }
    }
}

impl<Mods, D> Setup<D> for CustomModule<Mods>
where
    Mods: Setup<D>,
{
    fn setup(device: &mut D) -> custos::Result<()> {
        Mods::setup(device)
    }
}

fn main() {
    let _dev = CPU::<CustomModule<Base>>::new();
    // for actual usage, implement pass down traits / features
}
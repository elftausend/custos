use super::context::Context;
use crate::{Base, Module, Setup};
use std::rc::Rc;

pub struct Vulkan<Mods = Base> {
    pub modules: Mods,
    pub context: Rc<Context>,
}

impl<SimpleMods> Vulkan<SimpleMods> {
    #[inline]
    pub fn new<NewMods>(idx: usize) -> crate::Result<Vulkan<NewMods>>
    where
        SimpleMods: Module<Vulkan, Module = NewMods>,
        NewMods: Setup<Vulkan<NewMods>>,
    {
        let mut vulkan = Vulkan {
            modules: SimpleMods::new(),
            context: Rc::new(Context::new(idx)?),
        };
        NewMods::setup(&mut vulkan);
        Ok(vulkan)
    }
}

impl<Mods> Vulkan<Mods> {
    pub fn context(&self) -> Rc<Context> {
        self.context.clone()
    }
}

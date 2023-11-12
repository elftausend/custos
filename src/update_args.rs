use core::hash::BuildHasherDefault;
use std::collections::HashMap;

use crate::{HasId, NoHasher};

pub trait UpdateArgs {
    fn update_args(
        &mut self,
        buffers: &HashMap<crate::UniqueId, Box<dyn core::any::Any>, BuildHasherDefault<NoHasher>>,
    );
}

pub trait UpdateArg: HasId {
    fn update_arg(
        &mut self,
        buffers: &HashMap<crate::UniqueId, Box<dyn core::any::Any>, BuildHasherDefault<NoHasher>>,
    );
}

impl<T: UpdateArg + HasId> UpdateArgs for (T,) {
    fn update_args(
        &mut self,
        buffers: &HashMap<crate::UniqueId, Box<dyn core::any::Any>, BuildHasherDefault<NoHasher>>,
    ) {
        self.0.update_arg(buffers);
    }
}

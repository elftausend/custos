use core::hash::BuildHasherDefault;
use std::collections::HashMap;

use crate::{NoHasher, UniqueId};

pub trait UpdateArgs {
    fn update_args(
        &mut self,
        ids: &[Option<UniqueId>],
        buffers: &mut HashMap<
            crate::UniqueId,
            Box<dyn core::any::Any>,
            BuildHasherDefault<NoHasher>,
        >,
    ) -> crate::Result<()>;
}

pub trait UpdateArg {
    fn update_arg(
        &mut self,
        id: Option<UniqueId>,
        buffers: &mut HashMap<
            crate::UniqueId,
            Box<dyn core::any::Any>,
            BuildHasherDefault<NoHasher>,
        >,
    ) -> crate::Result<()>;
}

impl<T: UpdateArg> UpdateArgs for T {
    fn update_args(
        &mut self,
        ids: &[Option<UniqueId>],
        buffers: &mut HashMap<
            crate::UniqueId,
            Box<dyn core::any::Any>,
            BuildHasherDefault<NoHasher>,
        >,
    ) -> crate::Result<()> {
        self.update_arg(ids[0], buffers)
    }
}

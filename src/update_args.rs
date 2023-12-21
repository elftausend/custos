use core::hash::BuildHasherDefault;

#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;

use crate::{NoHasher, UniqueId};

/// A dummy trait for no-std context. [`UpdateArgs`] requires standard library code.
#[cfg(feature = "no-std")]
pub trait UpdateArgs {}

#[cfg(not(feature = "no-std"))]
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

/// A dummy trait for no-std context. [`UpdateArg`] requires standard library code.
#[cfg(feature = "no-std")]
pub trait UpdateArg {}

#[cfg(not(feature = "no-std"))]
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

#[cfg(not(feature = "no-std"))]
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

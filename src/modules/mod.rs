#[cfg(feature = "autograd")]
mod autograd;

#[cfg(feature = "autograd")]
pub use autograd::*;

mod base;
pub use base::*;

#[cfg(feature = "cached")]
mod cached;
#[cfg(feature = "cached")]
pub use cached::*;

#[cfg(feature = "graph")]
mod graph;
#[cfg(feature = "graph")]
pub use graph::*;

#[cfg(feature = "lazy")]
mod lazy;
#[cfg(feature = "lazy")]
pub use lazy::*;

#[cfg(feature = "fork")]
mod fork;
#[cfg(feature = "fork")]
pub use fork::*;

#[cfg(not(feature = "no-std"))]
use crate::{Buffer, Device, HasId, HashLocation, Id, ShallowCopy, Shape, UniqueId};
#[cfg(not(feature = "no-std"))]
use core::{any::Any, hash::BuildHasher};

#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;

#[cfg(not(feature = "no-std"))]
pub struct HashLocationCacheTrace {
    pub cache_idx: HashLocation<'static>,
    pub use_cache_idxs: Vec<HashLocation<'static>>,
}

#[cfg(not(feature = "no-std"))]
#[inline]
pub(crate) unsafe fn register_buf_any<T, D, S>(
    cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>,
    buf: &Buffer<T, D, S>,
) where
    T: 'static,
    D: Device + crate::IsShapeIndep + 'static,
    D::Data<T, S>: ShallowCopy,
    S: Shape,
{
    // shallow copy sets flag to AllocFlag::Wrapper

    let wrapped_data = buf.data.shallow();

    // let wrapped_data = D::convert::<T, S, T, S>(&buf.data, AllocFlag::Wrapper);

    let buf = Buffer {
        data: wrapped_data,
        device: buf.device,
    };
    let buf: Buffer<'static, T, D, S> = core::mem::transmute(buf);
    cache.insert(*buf.id(), Box::new(buf));
}

#[cfg(not(feature = "no-std"))]
#[inline]
pub fn unregister_buf_any(cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>, id: Id) {
    cache.remove(&id);
}

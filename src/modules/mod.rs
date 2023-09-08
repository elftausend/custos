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

mod graph;
pub use graph::*;

#[cfg(feature = "lazy")]
mod lazy;
#[cfg(feature = "lazy")]
pub use lazy::*;

#[cfg(feature = "fork")]
mod fork;
#[cfg(feature = "fork")]
pub use fork::*;

use core::{any::Any, hash::BuildHasher};
use std::collections::HashMap;
use crate::{UniqueId, Buffer, PtrConv, Device, Shape, flag::AllocFlag, Id, HasId};

#[inline]
pub(crate) unsafe fn register_buf<T, D, S>(
    cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>,
    buf: &Buffer<T, D, S>,
) where
    T: 'static,
    D: Device + PtrConv + 'static,
    S: Shape,
{
    let wrapped_data = D::convert::<T, S, T, S>(&buf.data, AllocFlag::Wrapper);
    let buf = Buffer {
        data: wrapped_data,
        device: buf.device,
    };
    let buf: Buffer<'static, T, D, S> = core::mem::transmute(buf);
    cache.insert(*buf.id(), Box::new(buf));
}

#[inline]
pub fn unregister_buf(cache: &mut HashMap<UniqueId, Box<dyn Any>, impl BuildHasher>, id: Id) {
    cache.remove(&id);
}

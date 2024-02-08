use core::hash::BuildHasher;
use std::collections::HashMap;

use crate::{BoxedShallowCopy, Buffer, Device, HasId, Id, ShallowCopy, Shape, UniqueId};

#[inline]
pub(crate) unsafe fn register_buf_copyable<T, D, S>(
    cache: &mut HashMap<UniqueId, Box<dyn BoxedShallowCopy>, impl BuildHasher>,
    buf: &Buffer<T, D, S>,
) where
    T: 'static,
    D: Device + crate::IsShapeIndep + 'static,
    D::Data<T, S>: ShallowCopy,
    S: Shape,
{
    // shallow copy sets flag to AllocFlag::Wrapper
    let wrapped_data = buf.data.shallow();

    let buf = Buffer {
        data: wrapped_data,
        device: buf.device,
    };
    let buf: Buffer<'static, T, D, S> = core::mem::transmute(buf);
    cache.insert(*buf.id(), Box::new(buf));
}

#[inline]
pub fn unregister_buf_copyable(
    cache: &mut HashMap<UniqueId, Box<dyn BoxedShallowCopy>, impl BuildHasher>,
    id: Id,
) {
    cache.remove(&id);
}

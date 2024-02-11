use core::hash::BuildHasherDefault;
use std::collections::HashMap;

use crate::{Cursor, HashLocation, LocationHasher};

pub type SpanStorage = HashMap<
    HashLocation<'static>,
    usize,
    BuildHasherDefault<LocationHasher>,
>;

#[inline]
pub fn span_update<D: Cursor>(
    dev: &D,
    location: (&'static str, u32, u32),
    span_storage: &mut SpanStorage,
) {
    let location = HashLocation::from(location);
    match span_storage.get(&location).copied() {
        Some(start) => unsafe { dev.set_cursor(start) },
        None => {
            span_storage.insert(location, dev.cursor());
        }
    };
}

#[macro_export]
macro_rules! span {
    ($dev:expr, $span_storage:expr) => {
        $crate::range::span_update(&$dev, (file!(), line!(), column!()), &mut $span_storage);
    };
}

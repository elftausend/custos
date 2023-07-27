use crate::Shape;

use super::{Alloc, Base, LocationId, CPU};

pub struct Buffer<'a, T = f32, D: Alloc = CPU<Base>, S: Shape = ()> {
    /// the type of pointer
    pub data: D::Data<T, S>,
    /// A reference to the corresponding device. Mainly used for operations without a device parameter.
    pub device: Option<&'a D>,
}

impl<'a, T, D: Alloc, S: Shape> Buffer<'a, T, D, S> {
    #[inline]
    pub fn new(device: &'a D, len: usize) -> Self {
        Buffer {
            data: device.alloc(len, crate::flag::AllocFlag::None),
            device: Some(device),
        }
    }
}

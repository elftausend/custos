use crate::Shape;

use super::{Alloc, Base, HasId, OnNewBuffer, CPU};

pub struct Buffer<'a, T = f32, D: Alloc = CPU<Base>, S: Shape = ()> {
    /// the type of pointer
    pub data: D::Data<T, S>,
    /// A reference to the corresponding device. Mainly used for operations without a device parameter.
    pub device: Option<&'a D>,
}

impl<'a, T, D: Alloc, S: Shape> Buffer<'a, T, D, S> {
    #[inline]
    pub fn new(device: &'a D, len: usize) -> Self
    where
        D: OnNewBuffer,
        D::Data<T, S>: HasId,
    {
        let buf = Buffer {
            data: device.alloc(len, crate::flag::AllocFlag::None),
            device: Some(device),
        };
        device.on_new_buffer(device, &buf);
        buf
    }

    #[inline]
    pub fn device(&self) -> &'a D {
        self.device
            .expect("Called device() on a deviceless buffer.")
    }
}

impl<'a, T, D: Alloc, S: Shape> HasId for Buffer<'a, T, D, S>
where
    D::Data<T, S>: HasId,
{
    #[inline]
    fn id(&self) -> super::Id {
        self.data.id()
    }
}

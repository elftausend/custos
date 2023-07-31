use crate::Shape;

use super::{Alloc, Base, Device, HasId, OnNewBuffer, CPU};

pub struct Buffer<'a, T = f32, D: Device = CPU<Base>, S: Shape = ()> {
    /// the type of pointer
    pub data: D::Data<T, S>,
    /// A reference to the corresponding device. Mainly used for operations without a device parameter.
    pub device: Option<&'a D>,
}

impl<'a, T, D: Device, S: Shape> Buffer<'a, T, D, S> {
    #[inline]
    pub fn new(device: &'a D, len: usize) -> Self
    where
        D: OnNewBuffer<T, D, S>,
    {
        let buf = Buffer {
            data: device.alloc(len, crate::flag::AllocFlag::None),
            device: Some(device),
        };

        // mind: on_new_buffer must be called for user buffers!
        device.on_new_buffer(device, &buf);
        buf
    }

    #[inline]
    pub fn device(&self) -> &'a D {
        self.device
            .expect("Called device() on a deviceless buffer.")
    }
}

impl<'a, T, D: Device, S: Shape> HasId for Buffer<'a, T, D, S> {
    #[inline]
    fn id(&self) -> super::Id {
        self.data.id()
    }
}

impl<'a, T, D: Device, S: Shape> Drop for Buffer<'a, T, D, S> {
    #[inline]
    fn drop(&mut self) {
        if let Some(device) = self.device {
            device.on_drop_buffer(device, self)
        }
    }
}

#[cfg(test)]
mod tests {}

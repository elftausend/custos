use crate::Shape;

use super::{Buffer, Device};

pub trait OnDropBuffer {
    fn on_drop_buffer<'a, T, D: Device, S: Shape>(&self, _device: &'a D, _buf: &Buffer<T, D, S>) {}
}

pub trait OnNewBuffer<T, D: Device, S: Shape> {
    fn on_new_buffer(&self, _device: &D, _new_buf: &Buffer<T, D, S>) {}
}

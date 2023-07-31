use crate::Shape;

use super::{Buffer, Device, Module};

pub trait OnDropBuffer {
    fn on_drop_buffer<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {}
}

pub trait OnNewBuffer<T, D: Device, S: Shape> {
    fn on_new_buffer(&self, _device: &D, _new_buf: &Buffer<T, D, S>) {}
}

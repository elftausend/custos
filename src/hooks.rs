use crate::Shape;

use super::{Buffer, Device};

pub trait OnDropBuffer {
    fn on_drop_buffer<T, D: Device, S: Shape>(&self, _device: &D, _buf: &Buffer<T, D, S>) {}
}

pub trait OnNewBuffer<T, D: Device, S: Shape> {
    #[track_caller]
    fn on_new_buffer(&self, _device: &D, _new_buf: &Buffer<T, D, S>) {}
}

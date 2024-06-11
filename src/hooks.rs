use crate::{Shape, Unit, WrappedData};

use super::{Buffer, Device};

pub trait OnDropBuffer: WrappedData {
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(&self, _device: &D, _buf: &Buffer<T, D, S>) {}
}

pub trait OnNewBuffer<T: Unit, D: Device, S: Shape = ()> {
    #[track_caller]
    fn on_new_buffer(&self, _device: &D, _new_buf: &Buffer<T, D, S>) {}
}

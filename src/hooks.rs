use crate::{Shape, Unit, WrappedData};

use super::{Buffer, Device};

pub trait OnDropBuffer: WrappedData {
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(&self, _device: &D, _buf: &Buffer<T, D, S>) {}
}

pub trait OnNewBuffer<'dev, T: Unit, D: Device, S: Shape = ()> {
    #[track_caller]
    fn on_new_buffer<'s>(&'s self, _device: &'dev D, _new_buf: &'s Buffer<'dev, T, D, S>) {}
}

pub trait OnNewBuffer2<'s, T: Unit, D: Device, S: Shape = ()> {
    #[track_caller]
    fn on_new_buffer(&'s self, _device: &D, _new_buf: &'s Buffer<T, D, S>) {}
}

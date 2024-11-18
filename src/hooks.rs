use crate::{Shape, Unit, WrappedData};

use super::{Buffer, Device};

pub trait OnDropBuffer: WrappedData {
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(&self, _device: &D, _buf: &Buffer<T, D, S>) {}
}

pub trait OnNewBuffer<'dev, T: Unit, D: Device, S: Shape = ()> {
    #[track_caller]
    unsafe fn on_new_buffer<'s>(
        &'dev self,
        _device: &'dev D,
        _new_buf: &'s mut Buffer<'dev, T, D, S>,
    ) {
    }
}

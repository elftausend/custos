use crate::{Shape, Unit, WrappedData};

use super::{Buffer, Device};

pub trait OnNewBuffer<'dev, T: Unit, D: Device, S: Shape = ()> {
    #[track_caller]
    unsafe fn on_new_buffer<'s>(
        &'dev self,
        _device: &'dev D,
        _new_buf: &'s mut Buffer<'dev, T, D, S>,
    ) {
    }
}

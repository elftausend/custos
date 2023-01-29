use crate::{Buffer, Device, Shape, CPU};

pub trait IsBuffer<'a, T, D: Device = CPU, S: Shape = ()> {
    fn buf(self) -> &'a Buffer<'a, T, D, S>;
}

impl<'a, T, D: Device, S: Shape> IsBuffer<'a, T, D, S> for &'a Buffer<'_, T, D, S> {
    fn buf(self) -> &'a Buffer<'a, T, D, S> {
        self
    }
}

impl<'a, T, D: Device, S: Shape> IsBuffer<'a, T, D, S> for &'a mut Buffer<'a, T, D, S> {
    fn buf(self) -> &'a Buffer<'a, T, D, S> {
        self
    }
}

impl<'a, T, D: Device, S: Shape> IsBuffer<'a, T, D, S> for Buffer<'a, T, D, S> {
    fn buf(self) -> &'a Buffer<'a, T, D, S> {
        todo!()
    }
}

pub trait IsBufferMut<'a, T, D: Device = CPU, S: Shape = ()> {
    fn buf_mut(&'a mut self) -> &mut Buffer<'a, T, D, S>;
}

impl<'a, T, D: Device, S: Shape> IsBufferMut<'a, T, D, S> for &'a mut Buffer<'a, T, D, S> {
    fn buf_mut(&'a mut self) -> &mut Buffer<'a, T, D, S> {
        self
    }
}

impl<'a, T, D: Device, S: Shape> IsBufferMut<'a, T, D, S> for Buffer<'a, T, D, S> {
    fn buf_mut(&mut self) -> &mut Buffer<'a, T, D, S> {
        self
    }
}

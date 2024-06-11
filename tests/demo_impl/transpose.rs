use custos::{Buffer, Device, Shape, Unit};

pub trait Transpose<T: Unit, D: Device, I: Shape = (), O: Shape = ()>: Device {
    fn transpose(&self, buf: Buffer<T, D, I>) -> Buffer<T, Self, O>;
}

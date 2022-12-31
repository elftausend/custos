use custos::{Buffer, Device, Shape};

pub trait Transpose<T, D: Device, I: Shape = (), O: Shape = ()>: Device {
    fn transpose(&self, buf: Buffer<T, D, I>) -> Buffer<T, Self, O>;
}

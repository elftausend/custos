use custos::{Buffer, Device};

mod cpu;
mod opencl;

pub trait ElementWise<T, D: Device, const N: usize = 0>: Device {
    fn add(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Self, N>;
    fn mul(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Self, N>;
}

use custos::{Buffer, Device};

mod cpu;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "wgpu")]
mod wgpu;

pub trait ElementWise<T, D: Device, const N: usize = 0>: Device {
    fn add(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Self, N>;
    fn mul(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Self, N>;
}

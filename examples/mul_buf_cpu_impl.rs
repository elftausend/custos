use custos::prelude::*;
use std::ops::{Deref, Mul};

pub trait MulBuf<T: Unit, S: Shape = (), D: Device = Self>: Sized + Device {
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
}

impl<Mods, T, S, D> MulBuf<T, S, D> for CPU<Mods>
where
    Mods: Retrieve<Self, T, S>,
    T: Unit + Mul<Output = T> + Copy + 'static,
    S: Shape,
    D: Device,
    D::Base<T, S>: Deref<Target = [T]>,
{
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs)).unwrap(); // unwrap or return error (update trait)

        for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(&mut out) {
            *out = *lhs * *rhs;
        }

        out
    }
}

fn main() {
    let device = CPU::<Base>::new();

    let lhs = Buffer::from((&device, &[-1, 2, 3, -4, 5, 9]));
    let rhs = Buffer::from((&device, &[4, -1, 7, 1, -2, 4]));

    let out = device.mul(&lhs, &rhs);
    assert_eq!(out.read(), [-4, -2, 21, -4, -10, 36]);
}

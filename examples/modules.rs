use std::ops::Add;

use custos::{Buffer, Device, MainMemory, Retrieve, Retriever, Shape, CPU};

pub trait ElementWise<T, D: Device, S: Shape>: Device {
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
}

pub fn ew_add_slice<T: Add<Output = T> + Copy>(lhs: &[T], rhs: &[T], out: &mut [T]) {
    for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(out) {
        *out = *lhs + *rhs;
    }
}

impl<T: Add<Output = T> + Copy, D: MainMemory, S: Shape, Mods: Retrieve<Self, T>>
    ElementWise<T, D, S> for CPU<Mods>
{
    #[track_caller]
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        ew_add_slice(lhs, rhs, &mut out);
        out
    }
}

fn main() {}

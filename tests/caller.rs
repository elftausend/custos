use std::{
    cell::{Cell, RefCell},
    ops::Add,
};

use custos::prelude::*;

#[derive(Debug, Default, Clone)]
pub struct Call {
    location: Cell<Option<&'static std::panic::Location<'static>>>,
}

impl Call {
    #[track_caller]
    pub fn call(&self) {
        self.location.set(Some(std::panic::Location::caller()))
    }
}

impl Add for &Call {
    type Output = Call;

    #[track_caller]
    fn add(self, rhs: Self) -> Self::Output {
        self.call();
        Call::default()
    }
}

pub fn add<'a, T: Add<Output = T> + Copy + 'static>(
    device: &'a CPU,
    lhs: &Buffer<T>,
    rhs: &Buffer<T>,
) -> Buffer<'a, T> {
    let len = std::cmp::min(lhs.len(), rhs.len());
    let mut out: Buffer<'_, T> = device.retrieve(len, (lhs, rhs));

    for idx in 0..len {
        out[idx] = lhs[idx] + rhs[idx];
    }

    out
}

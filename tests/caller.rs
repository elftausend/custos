use std::{
    cell::{Cell, RefCell},
    ops::Add,
};

use custos::{range, Buffer, CacheReturn, Device, CPU};

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

pub fn add<'a, T: Add<Output = T> + Copy>(
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

#[test]
fn test_caller() {
    let device = CPU::new();

    let lhs = device.buffer([1, 2, 3, 4]);
    let rhs = device.buffer([1, 2, 3, 4]);

    for _ in range(100) {
        add(&device, &lhs, &rhs);
    }

    assert_eq!(device.cache().nodes.len(), 3);

    for _ in 0..100 {
        add(&device, &lhs, &rhs);
    }

    assert_eq!(device.cache().nodes.len(), 102);

    let cell = RefCell::new(10);

    let x = cell.borrow();
    // cell.borrow_mut();

    let caller = Call::default();
    caller.call();

    let _ = &caller + &Call::default();

    let loc = caller.location;
    println!("location: {loc:?}");
}

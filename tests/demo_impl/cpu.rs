use custos::{prelude::Number, Buffer, Device, CPU, CPUCL};
use custos_macro::impl_stack;

#[cfg(feature = "stack-alloc")]
use custos::stack::Stack;

use super::ElementWise;

pub fn cpu_element_wise<T, F>(lhs: &[T], rhs: &[T], out: &mut [T], f: F)
where
    T: Copy,
    F: Fn(&mut T, T, T),
{
    let len = std::cmp::min(lhs.len(), rhs.len());

    for i in 0..len {
        f(&mut out[i], lhs[i], rhs[i])
    }
}

// TODO: write expansion example
#[impl_stack]
impl<T: Number, D> ElementWise<T, D> for CPU
where
    D: CPUCL,
{
    fn add(&self, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>) -> Buffer<T, CPU> {
        let mut out = self.retrieve(lhs.len, (lhs, rhs));
        cpu_element_wise(lhs, rhs, &mut out, |o, a, b| *o = a + b);
        out
    }

    fn mul(&self, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>) -> Buffer<T, CPU> {
        let mut out = self.retrieve(lhs.len, (lhs, rhs));
        cpu_element_wise(lhs, rhs, &mut out, |o, a, b| *o = a * b);
        out
    }
}

#[cfg(feature = "stack-alloc")]
#[test]
fn test_impl_stack() {
    let buf = Buffer::<i32, Stack, 5>::from([1, 2, 3, 4, 5]);
    let out = Stack.add(&buf, &buf);
    assert_eq!(out.as_slice(), &[2, 4, 6, 8, 10]);
}

/*
impl<T, const N: usize, D> ElementWise<T, D, N> for Stack
where
    D: CPUCL,
    Stack: Alloc<T, N>
{
    fn add(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Stack, N> {
        let len = std::cmp::min(lhs.len, rhs.len);
        let out = self.retrieve(len, (lhs, rhs));
        out
    }

    fn mul(&self, _lhs: &Buffer<T, D, N>, _rhs: &Buffer<T, D, N>) -> Buffer<T, Stack, N> {
        todo!()
    }
}*/

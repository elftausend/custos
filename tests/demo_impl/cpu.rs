use custos::{CPUCL, CPU, Buffer, Device, Alloc, stack::Stack, prelude::Number};

use super::ElementWise;

pub fn cpu_element_wise<T, F>(lhs: &[T], rhs: &[T], out: &mut [T], f: F) 
where 
    T: Copy, 
    F: Fn(&mut T, T, T)
{
    let len = std::cmp::min(lhs.len(), rhs.len());

    for i in 0..len {
        f(&mut out[i], lhs[i], rhs[i])
    }

}

//TODO stack macro: #[stack_impl]
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
}
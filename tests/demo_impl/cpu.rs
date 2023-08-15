use custos::{prelude::Number, Buffer, Device, Dim2, MainMemory, Shape, WithShape, CPU, Retriever};
use custos_macro::impl_stack;
//use custos_macro::impl_stack;

#[cfg(feature = "stack")]
use custos::Stack;

use super::{transpose::Transpose, ElementWise};

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
impl<T, D, S> ElementWise<T, D, S> for CPU
where
    T: Number,
    D: MainMemory,
    S: Shape,
{
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, CPU, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        cpu_element_wise(lhs, rhs, &mut out, |o, a, b| *o = a + b);
        out
    }

    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, CPU, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        cpu_element_wise(lhs, rhs, &mut out, |o, a, b| *o = a * b);
        out
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn test_const_size_buf() {
    let device = CPU::<Base>::new();

    let lhs = Buffer::with(&device, [1, 2, 3, 4]);
    let rhs = Buffer::with(&device, [4, 1, 9, 4]);

    //let lhs = Buffer::from((&device, [1, 2, 3, 4]));
    //let rhs = Buffer::from((&device, [4, 1, 9, 4]));

    device.add(&lhs, &rhs);

    let device = custos::WGPU::new(wgpu::Backends::all()).unwrap();

    let lhs = Buffer::with(&device, [1, 2, 3, 4]);
    let rhs = Buffer::with(&device, [4, 1, 9, 4]);

    device.add(&lhs, &rhs);

    let device = custos::OpenCL::new(0).unwrap();

    let lhs = Buffer::with(&device, [1, 2, 3, 4]);
    let rhs = Buffer::with(&device, [4, 1, 9, 4]);

    device.add(&lhs, &rhs);
}

#[cfg(feature = "cpu")]
#[cfg(feature = "stack")]
#[test]
fn test_impl_stack() {
    use custos::{Dim1, Base};

    let device = CPU::<Base>::new();
    let buf = Buffer::<i32, _>::from((&device, [1, 2, 3, 4, 5]));
    let out = device.add(&buf, &buf);
    assert_eq!(out.as_slice(), &[2, 4, 6, 8, 10]);

    let stack = Stack::new();

    let buf = Buffer::<i32, Stack, Dim1<5>>::with(&stack, [1, 2, 3, 4, 5]);
    let out = stack.add(&buf, &buf);
    assert_eq!(out.as_slice(), &[2, 4, 6, 8, 10]);
}

/*
impl<T, const N: usize, D> ElementWise<T, D, N> for Stack
where
    T: Number,
    D: MainMemory,
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
*/

#[cfg(feature = "stack")]
impl<T, D: Device, const A: usize, const B: usize> Transpose<T, D, Dim2<A, B>, Dim2<B, A>>
    for Stack
{
    fn transpose(&self, _buf: Buffer<T, D, Dim2<A, B>>) -> Buffer<T, Self, Dim2<B, A>> {
        todo!()
    }
}

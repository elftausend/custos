//! The Stack module provides the Stack backend for custos.

mod impl_buffer;
mod stack_device;

use core::ops::{AddAssign, Deref, DerefMut};

pub use stack_device::*;

use crate::{
    cpu_stack_ops::clear_slice, ApplyFunction, Buffer, ClearBuf, Device, Eval, MayToCLSource,
    OnDropBuffer, Resolve, Retrieve, Retriever, Shape, ToVal, UnaryGrad, ZeroGrad,
};

// #[impl_stack]
impl<T, D, S> ClearBuf<T, S, D> for Stack
where
    T: Default,
    D: Device,
    D::Base<T, S>: DerefMut<Target = [T]>,
    S: Shape,
{
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, D, S>) {
        clear_slice(buf)
    }
}

impl<Mods, T> ZeroGrad<T> for Stack<Mods>
where
    T: Default,
    Mods: OnDropBuffer,
{
    #[inline]
    fn zero_grad<S: Shape>(&self, data: &mut Self::Base<T, S>) {
        clear_slice(data)
    }
}

impl<Mods, T, D, S> ApplyFunction<T, S, D> for Stack<Mods>
where
    Mods: Retrieve<Self, T, S>,
    T: Copy + Default + ToVal + 'static,
    D: Device,
    D::Base<T, S>: Deref<Target = [T]>,
    S: Shape,
{
    fn apply_fn<F>(&self, buf: &Buffer<T, D, S>, f: impl Fn(Resolve<T>) -> F) -> Buffer<T, Self, S>
    where
        F: Eval<T> + MayToCLSource,
    {
        let mut out = self.retrieve(buf.len(), buf);

        crate::cpu_stack_ops::apply_fn_slice(buf, &mut out, f);

        out
    }
}

impl<Mods, T, D, S> UnaryGrad<T, S, D> for Stack<Mods>
where
    Mods: OnDropBuffer,
    T: AddAssign + Copy + core::ops::Mul<Output = T>,
    S: Shape,
    D: Device,
    D::Base<T, S>: Deref<Target = [T]> + DerefMut,
{
    #[inline]
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F,
    ) where
        F: Eval<T> + MayToCLSource,
    {
        crate::cpu_stack_ops::add_unary_grad(lhs, out, lhs_grad, lhs_grad_fn)
    }
}

#[cfg(feature = "cpu")]
#[cfg(test)]
mod tests {
    use crate::{Alloc, Base, Buffer, Device, Dim1, Retrieve, Retriever, Shape, CPU};
    use core::ops::{Add, Deref};

    use super::stack_device::Stack;

    pub trait AddBuf<T, D: Device = Self, S: Shape = ()>: Device {
        fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
    }

    /*// Without stack support
    impl<T, D> AddBuf<T, D> for CPU
    where
        T: Add<Output = T> + Clone,
        D: CPUCL,
    {
        fn add(&self, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>) -> Buffer<T, Self> {
            let len = core::cmp::min(lhs.len, rhs.len);

            let mut out = self.retrieve(len, (lhs, rhs));
            for i in 0..len {
                out[i] = lhs[i].clone() + rhs[i].clone();
            }
            out
        }
    }*/

    impl<Mods: Retrieve<Self, T>, T, D> AddBuf<T, D> for CPU<Mods>
    where
        D: Device,
        D::Base<T, ()>: Deref<Target = [T]>,
        T: Add<Output = T> + Clone,
    {
        fn add(&self, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>) -> Buffer<T, Self> {
            let len = core::cmp::min(lhs.len(), rhs.len());

            let mut out = self.retrieve(len, (lhs, rhs));
            for i in 0..len {
                out[i] = lhs[i].clone() + rhs[i].clone();
            }
            out
        }
    }

    impl<T, D, S: Shape> AddBuf<T, D, S> for Stack
    where
        Stack: Alloc<T>,
        D: Device,
        D::Base<T, S>: Deref<Target = [T]>,
        T: Add<Output = T> + Copy + Default,
    {
        fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
            let mut out = self.retrieve(S::LEN, (lhs, rhs));

            for i in 0..S::LEN {
                out[i] = lhs[i] + rhs[i];
            }
            out
        }
    }

    /*#[test]
    fn test_stack_from_borrowed() {
        let device = Stack;

        // TODO: fix stack from_const while borrowed
        let mut buf = Buffer::from((&device, [1., 2., 3., 4., 5., 6.]));


        assert_eq!(buf.read(), [1., 2., 3., 4., 5., 6.,]);
        buf.clear();
        assert_eq!(buf.read(), [0.; 6]);
    }*/

    #[test]
    fn test_stack() {
        let dev = Stack::new();

        // TODO fix stack from_const while borrowed
        //let buf = Buffer::<f32, Stack, _>::from((&Stack, [1f32; 100]));
        let buf = Buffer::<f32, Stack, Dim1<100>>::from((&dev, [1f32; 100]));

        let out = dev.add(&buf, &buf);
        assert_eq!(out.data.array, [2.; 100]);

        let cpu = CPU::<Base>::new();

        // implement Buffer::<f32, _, 100> for cpu?
        //let buf = Buffer::<f32>::new(&cpu, 100);
        let buf = Buffer::from((&cpu, [1f32; 100]));
        let out = cpu.add(&buf, &buf);
        assert_eq!(&**out, &[2.; 100]);
    }
}

mod impl_buffer;
mod stack_array;
mod stack_device;

pub use stack_array::StackArray;
pub use stack_device::*;

#[cfg(feature = "cpu")]
#[cfg(test)]
mod tests {
    use crate::{Alloc, Buffer, Device, Dim1, MainMemory, Shape, CPU};
    use core::ops::Add;

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

    impl<T, D> AddBuf<T, D> for CPU<'_>
    where
        D: MainMemory,
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
        for<'a> Stack: Alloc<'a, T, S>,
        D: MainMemory,
        T: Add<Output = T> + Clone,
    {
        fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
            let mut out = self.retrieve(S::LEN, (lhs, rhs));

            for i in 0..S::LEN {
                out[i] = lhs[i].clone() + rhs[i].clone();
            }
            out
        }
    }

    #[test]
    fn test_stack() {
        let buf = Buffer::<f32, Stack, Dim1<100>>::from((Stack, [1f32; 100]));

        let out = Stack.add(&buf, &buf);
        assert_eq!(out.ptr.array, [2.; 100]);

        let cpu = CPU::new();

        // implement Buffer::<f32, _, 100> for cpu?
        //let buf = Buffer::<f32>::new(&cpu, 100);
        let buf = Buffer::from((&cpu, [1f32; 100]));
        let out = cpu.add(&buf, &buf);
        assert_eq!(&*out, &[2.; 100]);
    }
}

mod buffer_impl;
mod stack_device;

#[cfg(test)]
mod tests {
    use std::ops::Add;
    use crate::{Alloc, Buffer, Device, CPU, CPUCL, IsCPU};

    use super::stack_device::Stack;

    pub trait AddBuf<T, D: Device, const N: usize = 0>: Device {
        fn add(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Self, N>;
    }

    
    /*// Without stack support
    impl<T, D> AddBuf<T, D> for CPU
    where 
        T: Add<Output = T> + Clone,
        D: CPUCL,
    {
        fn add(&self, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>) -> Buffer<T, Self> {
            let len = std::cmp::min(lhs.len, rhs.len);

            let mut out = self.retrieve(len, (lhs, rhs));
            for i in 0..len {
                out[i] = lhs[i].clone() + rhs[i].clone();
            }
            out
        }
    }*/


    impl<IsCpu, const N: usize, T, D> AddBuf<T, D, N> for IsCpu
    where
        IsCpu: IsCPU + Alloc<T, N>,
        D: CPUCL,
        T: Add<Output = T> + Clone,
    {
        fn add(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Self, N> {
            let len = std::cmp::min(lhs.len, rhs.len);

            let mut out = self.retrieve(len, (lhs, rhs));
            for i in 0..len {
                out[i] = lhs[i].clone() + rhs[i].clone();
            }
            out
        }
    }
    

    #[test]
    fn test_stack() {
        let stack = Stack;

        let mut buf = Buffer::<f32, _, 100>::new(&stack, 0);
        buf.copy_from_slice(&[1.; 100]);

        let out = stack.add(&buf, &buf);
        assert_eq!(out.as_slice(), &[2.; 100]);

        let cpu = CPU::new();
        
        // implement Buffer::<f32, _, 100> for cpu?
        let buf = Buffer::<f32>::new(&cpu, 100);
        cpu.add(&buf, &buf);
    }
}

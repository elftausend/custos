mod stack_device;
mod buffer_impl;


#[cfg(test)]
mod tests {
    use crate::{Buffer, Device, CPU, CPUCL, CacheAble, Alloc};

    use super::stack_device::Stack;

    pub trait AddBuf<T, D: Device, const N: usize = 0>: Device {
        fn add(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Self, N>;
    }

    /*
    // Without stack support
    impl<T, D: CPUCL> AddBuf<T, D> for D { 
        fn add(&self, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>) -> Buffer<T, Self> {
            todo!()
        }
    }*/

    impl<const N: usize, T, D: CPUCL + Alloc<T, N>> AddBuf<T, D, N> for D { 
        fn add(&self, lhs: &Buffer<T, D, N>, rhs: &Buffer<T, D, N>) -> Buffer<T, Self, N> {
            let len = std::cmp::min(lhs.len, rhs.len);
            
            let out = self.retrieve(len, (lhs.node.idx, rhs.node.idx));
            for i in 0..len {
                    
            }
            out
        }
    }

    #[test]
    fn test_stack() {
        let stack = Stack;

        let buf = Buffer::<f32, _, 100>::new(&stack, 100);
        stack.add(&buf, &buf);

        let cpu = CPU::new();
        let buf = Buffer::<f32, _, 0>::new(&cpu, 100);
        cpu.add(&buf, &buf);
    }
}
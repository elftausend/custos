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
            let out: Buffer<T, D, N> = D::Cache::retrieve(self, N, (lhs.node.idx, rhs.node.idx));
            for i in 0..N {
                
            }
            out
        }
    }

    #[test]
    fn test_stack() {
        let stack = Stack;

        let buf = Buffer::<f32, _, 100>::new(&stack, 100);
    }
}
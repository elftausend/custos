use crate::{Alloc, Device, DevicelessAble, CPUCL, Buffer};

use super::stack_array::StackArray;

#[derive(Debug, Clone, Copy)]
pub struct Stack;



impl<T: Copy + Default> DevicelessAble<T> for Stack {}

impl Device for Stack {
    type Ptr<U, const N: usize> = StackArray<N, U>;
    type Cache<const N: usize> = ();
}

impl CPUCL for Stack {
    #[inline]
    fn buf_as_slice<'a, T, const N: usize>(buf: &'a Buffer<T, Self, N>) -> &'a [T] {
        &buf.ptr.array
    }

    #[inline]
    fn buf_as_slice_mut<'a, T, const N: usize>(buf: &'a mut Buffer<T, Self, N>) -> &'a mut [T] {
        &mut buf.ptr.array
    }
}

impl<const N: usize, T: Copy + Default> Alloc<T, N> for Stack {
    #[inline]
    fn alloc(&self, _len: usize) -> StackArray<N, T> {
        // TODO: one day... use const expressions
        if N == 0 {
            panic!("The size (N) of a stack allocated buffer must be greater than 0.");
        }
        StackArray {
            array: [T::default(); N],
        }
    }

    #[inline]
    fn with_slice(&self, data: &[T]) -> StackArray<N, T> {
        let mut array = self.alloc(0);
        array.copy_from_slice(&data[..N]);
        array
    }

    #[inline]
    fn with_array(&self, array: [T; N]) -> <Self as Device>::Ptr<T, N> {
        StackArray { array }
    }
}

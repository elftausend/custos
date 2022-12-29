use crate::{Alloc, Buffer, Device, DevicelessAble, MainMemory, Read, shape::Shape};

use super::stack_array::StackArray;

#[derive(Debug, Clone, Copy)]
pub struct Stack;

impl<'a, T: Copy + Default> DevicelessAble<'a, T> for Stack {}

impl Device for Stack {
    type Ptr<U, S: Shape> = S::ARR<U>;
    type Cache = ();

    fn new() -> crate::Result<Self> {
        Ok(Stack)
    }
}

impl MainMemory for Stack {
    #[inline]
    fn buf_as_slice<'a, T, S: Shape>(buf: &'a Buffer<T, Self, S>) -> &'a [T] {
        &buf.ptr.array
    }

    #[inline]
    fn buf_as_slice_mut<'a, T, S: Shape>(buf: &'a mut Buffer<T, Self, S>) -> &'a mut [T] {
        &mut buf.ptr.array
    }
}

impl<'a, S: Shape, T: Copy + Default> Alloc<'a, T, S> for Stack {
    #[inline]
    fn alloc(&self, _len: usize) -> StackArray<N, T> {
        // TODO: one day... use const expressions
        if S::LEN == 0 {
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

impl<T: Clone, const N: usize> Read<T, Stack, N> for Stack {
    type Read<'a> = [T; N]
    where
        T: 'a,
        Stack: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, Stack, N>) -> Self::Read<'a> {
        buf.ptr.array.clone()
    }

    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn read_to_vec(&self, buf: &Buffer<T, Stack, N>) -> Vec<T>
    where
        T: Default,
    {
        buf.ptr.to_vec()
    }
}

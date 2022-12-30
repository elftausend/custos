use crate::{shape::Shape, Alloc, Buffer, Device, DevicelessAble, MainMemory, Read};

use super::stack_array::StackArray;

#[derive(Debug, Clone, Copy)]
pub struct Stack;

impl<'a, T: Copy + Default, S: Shape> DevicelessAble<'a, T, S> for Stack {}

impl Device for Stack {
    type Ptr<U, S: Shape> = StackArray<S, U>;
    type Cache = ();

    fn new() -> crate::Result<Self> {
        Ok(Stack)
    }
}

impl MainMemory for Stack {
    #[inline]
    fn buf_as_slice<'a, T, S: Shape>(buf: &'a Buffer<T, Self, S>) -> &'a [T] {
        &buf.ptr
    }

    #[inline]
    fn buf_as_slice_mut<'a, T, S: Shape>(buf: &'a mut Buffer<T, Self, S>) -> &'a mut [T] {
        &mut buf.ptr
    }
}

impl<'a, S: Shape, T: Copy + Default> Alloc<'a, T, S> for Stack {
    #[inline]
    fn alloc(&self, _len: usize) -> StackArray<S, T> {
        // TODO: one day... use const expressions
        if S::LEN == 0 {
            panic!("The size (N) of a stack allocated buffer must be greater than 0.");
        }
        StackArray {
            array: <S as Shape>::new(),
        }
    }

    #[inline]
    fn with_slice(&self, data: &[T]) -> Self::Ptr<T, S> {
        let mut array: StackArray<S, T> = <Stack as Alloc<'_, T, S>>::alloc(self, 0);
        unsafe {
            array.flatten_mut().copy_from_slice(&data[..S::LEN]);
        }
        array
    }

    /* TODO
    #[inline]
    fn with_array<const N: usize>(&self, array: [T; N]) -> Self::Ptr<T, S> {
        //StackArray { array }
        todo!()
    }*/
}

impl<T: Copy, S: Shape> Read<T, Stack, S> for Stack
where
    S::ARR<T>: Clone,
{
    type Read<'a> = S::ARR<T>
    where
        T: 'a,
        Stack: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, Stack, S>) -> Self::Read<'a> {
        buf.ptr.array.clone()
    }

    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn read_to_vec(&self, buf: &Buffer<T, Stack, S>) -> Vec<T>
    where
        T: Default,
    {
        unsafe { buf.ptr.flatten().to_vec() }
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape::Dim2, Buffer, Stack};

    #[test]
    fn test_dim2() {
        let buf = Buffer::<f64, Stack, Dim2<2, 3>>::from((&Stack, &[3., 2., 1., 4., 7., 1.,]));

        for val in buf.iter() {
            println!("val: {val}");
        }

        assert_eq!(buf.read(), [[3., 2., 1.,], [4., 7., 1.]]);

        assert_eq!(buf.read_to_vec(), [3., 2., 1., 4., 7., 1.]);

    }
}

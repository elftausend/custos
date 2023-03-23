use crate::{
    flag::AllocFlag, shape::Shape, Alloc, Buffer, CloneBuf, Device, DevicelessAble, MainMemory,
    Read, StackArray, WriteBuf,
};

/// A device that allocates memory on the stack.
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
    fn as_ptr<T, S: Shape>(ptr: &Self::Ptr<T, S>) -> *const T {
        ptr.as_ptr()
    }

    #[inline]
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Ptr<T, S>) -> *mut T {
        ptr.as_ptr_mut()
    }
}

impl<'a, S: Shape, T: Copy + Default> Alloc<'a, T, S> for Stack {
    #[inline]
    fn alloc(&self, _len: usize, _flag: AllocFlag) -> StackArray<S, T> {
        StackArray::new()
    }

    #[inline]
    fn with_slice(&self, data: &[T]) -> Self::Ptr<T, S> {
        let mut array: StackArray<S, T> =
            <Stack as Alloc<'_, T, S>>::alloc(self, 0, AllocFlag::None);
        array.flatten_mut().copy_from_slice(&data[..S::LEN]);

        array
    }

    #[inline]
    fn with_array(&'a self, array: <S as Shape>::ARR<T>) -> <Self as Device>::Ptr<T, S>
    where
        T: Clone,
    {
        StackArray::from_array(array)
    }
}

/*impl GraphReturn for Stack {
    fn graph(&self) -> core::cell::RefMut<crate::Graph> {
        unimplemented!()
    }
}*/

impl<T: Copy, S: Shape> Read<T, S> for Stack
where
    S::ARR<T>: Copy,
{
    type Read<'a> = S::ARR<T>
    where
        T: 'a,
        Stack: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, Stack, S>) -> Self::Read<'a> {
        buf.ptr.array
    }

    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn read_to_vec(&self, buf: &Buffer<T, Stack, S>) -> Vec<T>
    where
        T: Default,
    {
        buf.ptr.flatten().to_vec()
    }
}

impl<'a, T, S: Shape> CloneBuf<'a, T, S> for Stack
where
    <Stack as Device>::Ptr<T, S>: Copy,
{
    fn clone_buf(&'a self, buf: &Buffer<'a, T, Self, S>) -> Buffer<'a, T, Self, S> {
        Buffer {
            ptr: buf.ptr,
            device: Some(&Stack),
            ident: buf.ident,
            requires_grad: false,
        }
    }
}

impl<T: Copy, S: Shape> WriteBuf<T, S> for Stack {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, Self, S>, data: &[T]) {
        buf.copy_from_slice(data)
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self, S>, src: &Buffer<T, Self, S>) {
        self.write(dst, src)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no-std"))]
    use crate::{shape::Dim2, Buffer, Stack};

    #[cfg(not(feature = "no-std"))]
    #[test]
    fn test_dim2() {
        let buf = Buffer::<f64, Stack, Dim2<2, 3>>::from((&Stack, &[3., 2., 1., 4., 7., 1.]));

        for val in buf.iter() {
            println!("val: {val}");
        }

        assert_eq!(buf.read(), [[3., 2., 1.,], [4., 7., 1.]]);

        assert_eq!(buf.read_to_vec(), [3., 2., 1., 4., 7., 1.]);
    }
}

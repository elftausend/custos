use core::convert::Infallible;

use crate::{
    flag::AllocFlag, impl_buffer_hook_traits, impl_retriever, shape::Shape, Alloc, Base, Buffer,
    CloneBuf, Device, DevicelessAble, OnDropBuffer, Read, StackArray, WriteBuf,
};

/// A device that allocates memory on the stack.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stack<Mods = Base> {
    modules: Mods,
}

impl Stack {
    pub fn new() -> Stack<Base> {
        Stack { modules: Base }
    }
}

impl_buffer_hook_traits!(Stack);
impl_retriever!(Stack, Copy + Default);

impl<'a, T: Copy + Default, S: Shape> DevicelessAble<'a, T, S> for Stack {}

impl<Mods: OnDropBuffer> Device for Stack<Mods> {
    type Data<U, S: Shape> = StackArray<S, U>;
    type Error = Infallible;

    fn new() -> Result<Self, Infallible> {
        todo!()
    }
}

impl<Mods: OnDropBuffer, T: Copy + Default> Alloc<T> for Stack<Mods> {
    #[inline]
    fn alloc<S: Shape>(&self, _len: usize, _flag: AllocFlag) -> StackArray<S, T> {
        StackArray::new()
    }

    #[inline]
    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> Self::Data<T, S> {
        let mut array: StackArray<S, T> =
            <Stack<Mods> as Alloc<T>>::alloc(self, 0, AllocFlag::None);
        array.flatten_mut().copy_from_slice(&data[..S::LEN]);

        array
    }

    #[inline]
    fn alloc_from_array<S: Shape>(
        &self,
        array: <S as Shape>::ARR<T>,
    ) -> <Self as Device>::Data<T, S>
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
        buf.data.array
    }

    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn read_to_vec(&self, buf: &Buffer<T, Stack, S>) -> Vec<T>
    where
        T: Default,
    {
        buf.data.flatten().to_vec()
    }
}

impl<'a, T, S: Shape> CloneBuf<'a, T, S> for Stack
where
    <Stack as Device>::Data<T, S>: Copy,
{
    fn clone_buf(&'a self, buf: &Buffer<'a, T, Self, S>) -> Buffer<'a, T, Self, S> {
        Buffer {
            data: buf.data,
            device: Some(self),
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

#[cfg(feature = "autograd")]
impl<Mods: crate::TapeActions> crate::TapeActions for Stack<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<crate::Tape>> {
        None
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<crate::Tape>> {
        None
    }
}
#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no-std"))]
    use crate::{shape::Dim2, Buffer, Stack};

    #[cfg(not(feature = "no-std"))]
    #[test]
    fn test_dim2() {
        let dev = Stack::new();
        let buf = Buffer::<f64, Stack, Dim2<2, 3>>::from((&dev, &[3., 2., 1., 4., 7., 1.]));

        for val in buf.iter() {
            println!("val: {val}");
        }

        assert_eq!(buf.read(), [[3., 2., 1.,], [4., 7., 1.]]);

        assert_eq!(buf.read_to_vec(), [3., 2., 1., 4., 7., 1.]);
    }
}

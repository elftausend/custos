use core::convert::Infallible;

use crate::{
    AddOperationPassDown, Alloc, Base, Buffer, CloneBuf, Device, DeviceError, DevicelessAble, Read,
    StackArray, Unit, WrappedData, WriteBuf, flag::AllocFlag, impl_buffer_hook_traits,
    impl_retriever, impl_wrapped_data, pass_down_cursor, pass_down_grad_fn, pass_down_tape_actions,
    pass_down_use_gpu_or_cpu, shape::Shape,
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
impl_wrapped_data!(Stack);
pass_down_cursor!(Stack);
pass_down_grad_fn!(Stack);
pass_down_tape_actions!(Stack);
pass_down_use_gpu_or_cpu!(Stack);
#[cfg(feature = "graph")]
crate::pass_down_optimize_mem_graph!(Stack);
impl<Mods> AddOperationPassDown for Stack<Mods> {}

impl<'a, T: Unit + Copy + Default, S: Shape> DevicelessAble<'a, T, S> for Stack {}

impl<Mods: WrappedData> Device for Stack<Mods> {
    type Data<'a, U: Unit, S: Shape> = Self::Wrap<'a, U, Self::Base<U, S>>;
    type Base<T: Unit, S: Shape> = StackArray<S, T>;
    type Error = Infallible;

    fn new() -> Result<Self, Infallible> {
        todo!()
    }

    #[inline]
    fn wrap_to_data<'a, T: Unit, S: Shape>(
        &self,
        wrap: Self::Wrap<'a, T, Self::Base<T, S>>,
    ) -> Self::Data<'a, T, S> {
        wrap
    }

    #[inline]
    fn data_as_wrap<'a, 'b, T: Unit, S: Shape>(
        data: &'b Self::Data<'a, T, S>,
    ) -> &'b Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }

    #[inline]
    fn data_as_wrap_mut<'a, 'b, T: Unit, S: Shape>(
        data: &'b mut Self::Data<'a, T, S>,
    ) -> &'b mut Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }

    #[inline]
    fn default_base_to_data<'a, T: Unit, S: Shape>(
        &'a self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.wrap_in_base(base)
    }

    fn default_base_to_data_unbound<'a, T: Unit, S: Shape>(
        &self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.wrap_in_base_unbound(base)
    }
}

impl<Mods: WrappedData, T: Unit + Copy + Default> Alloc<T> for Stack<Mods> {
    #[inline]
    fn alloc<S: Shape>(&self, _len: usize, _flag: AllocFlag) -> crate::Result<StackArray<S, T>> {
        Ok(StackArray::new())
    }

    #[inline]
    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>> {
        if data.len() < S::LEN {
            return Err(DeviceError::ShapeLengthMismatch.into());
        }
        let mut array = Alloc::<T>::alloc(self, 0, AllocFlag::None)?;
        array.flatten_mut().copy_from_slice(&data[..S::LEN]);

        Ok(array)
    }

    #[inline]
    fn alloc_from_array<S: Shape>(
        &self,
        array: <S as Shape>::Array<T>,
    ) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        Ok(StackArray::from_array(array))
    }
}

impl<T: Unit + Copy, S: Shape> Read<T, S> for Stack
where
    S::Array<T>: Copy,
{
    type Read<'a>
        = S::Array<T>
    where
        T: 'a,
        Stack: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Self::Base<T, S>) -> Self::Read<'a>
    where
        Stack: 'a,
    {
        buf.array
    }

    #[inline]
    #[cfg(feature = "std")]
    fn read_to_vec(&self, buf: &Self::Base<T, S>) -> Vec<T>
    where
        T: Default,
    {
        buf.flatten().to_vec()
    }
}

impl<'a, T: Unit, S: Shape> CloneBuf<'a, T, S> for Stack
where
    <Stack as Device>::Data<'a, T, S>: Copy,
{
    #[inline]
    fn clone_buf(&'a self, buf: &Buffer<'a, T, Self, S>) -> Buffer<'a, T, Self, S> {
        Buffer {
            data: buf.data,
            device: Some(self),
        }
    }
}

impl<T: Unit + Copy, S: Shape> WriteBuf<T, S> for Stack {
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
    #[cfg(feature = "std")]
    use crate::{Buffer, Stack, shape::Dim2};

    #[cfg(feature = "std")]
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

use core::{
    convert::Infallible,
    ops::{Deref, DerefMut},
};

use crate::{
    flag::AllocFlag, Alloc, Buffer, CloneBuf, Device, HasId, IsBasePtr, OnDropBuffer, PtrType,
    ShallowCopy, Unit, WrappedData,
};

#[derive(Debug, Default)]
/// Makes it possible to use a single number in a [`Buffer`].
pub struct Num<T> {
    /// The stored number.
    pub num: T,
}

impl<T: Unit> PtrType for Num<T> {
    #[inline]
    fn size(&self) -> usize {
        0
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        crate::flag::AllocFlag::None
    }

    #[inline]
    unsafe fn set_flag(&mut self, _flag: AllocFlag) {}
}

impl<T> HasId for Num<T> {
    fn id(&self) -> crate::Id {
        todo!()
    }
}

impl<T> From<T> for Num<T> {
    #[inline]
    fn from(num: T) -> Self {
        Num { num }
    }
}

impl<T> ShallowCopy for Num<T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        unimplemented!()
    }
}

impl Device for () {
    type Data<'a, T: Unit, S: crate::Shape> = Self::Base<T, S>;
    type Base<T: Unit, S: crate::Shape> = Num<T>;

    type Error = Infallible;

    fn new() -> Result<Self, Infallible> {
        Ok(())
    }

    #[inline(always)]
    fn default_base_to_data<'a, T: Unit, S: crate::Shape>(
        &'a self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        base
    }

    #[inline(always)]
    fn default_base_to_data_unbound<'a, T: Unit, S: crate::Shape>(
        &self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        base
    }

    #[inline(always)]
    fn wrap_to_data<'a, T: Unit, S: crate::Shape>(
        &self,
        wrap: Self::Wrap<'a, T, Self::Base<T, S>>,
    ) -> Self::Data<'a, T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<'a, 'b, T: Unit, S: crate::Shape>(
        data: &'b Self::Data<'a, T, S>,
    ) -> &'b Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }

    fn data_as_wrap_mut<'a, 'b, T: Unit, S: crate::Shape>(
        data: &'b mut Self::Data<'a, T, S>,
    ) -> &'b mut Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }
}

impl<T: Unit + Default> Alloc<T> for () {
    #[inline]
    fn alloc<S: crate::Shape>(
        &self,
        _len: usize,
        _flag: AllocFlag,
    ) -> crate::Result<Self::Base<T, S>> {
        Ok(Num::default())
    }

    #[inline]
    fn alloc_from_slice<S: crate::Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        Ok(data[0].clone().into())
    }
}

impl WrappedData for () {
    type Wrap<'a, T: Unit, Base: IsBasePtr> = Base;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: IsBasePtr>(&'a self, base: Base) -> Self::Wrap<'a, T, Base> {
        base
    }

    #[inline]
    fn wrap_in_base_unbound<'a, T: Unit, Base: IsBasePtr>(
        &self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        base
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base {
        wrap
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
        wrap
    }
}

impl OnDropBuffer for () {}

impl<'a, T: Unit + Clone> CloneBuf<'a, T> for () {
    #[inline]
    fn clone_buf(&self, buf: &Buffer<'a, T, Self>) -> Buffer<'a, T, Self> {
        Buffer {
            data: Num {
                num: buf.data.num.clone(),
            },
            device: buf.device,
        }
    }
}

impl<T: crate::number::Number> From<T> for Buffer<'_, T, ()> {
    #[inline]
    fn from(ptr: T) -> Self {
        Buffer {
            data: Num { num: ptr },
            device: None,
        }
    }
}

impl<'a, T: Unit> Buffer<'a, T, ()> {
    /// A [`Num`] [`Buffer`] is safe to copy.
    /// This method returns a new "[`Buffer`]" with the same single value.
    #[inline]
    pub fn copy(&self) -> Self
    where
        T: Unit + Copy,
    {
        Buffer {
            data: Num { num: self.data.num },
            device: self.device,
        }
    }

    /// Used if the `Buffer` contains only a single value.
    /// By derefencing this `Buffer`, you obtain this value as well (which is probably preferred).
    ///
    /// # Example
    ///
    /// ```
    /// use custos::Buffer;
    ///
    /// let x: Buffer<f32, _> = 7f32.into();
    /// assert_eq!(**x, 7.);
    /// assert_eq!(x.item(), &7.);
    ///
    /// ```
    #[inline]
    pub fn item(&self) -> &T
    where
        T: Unit,
    {
        &self.data.num
    }

    /// Used if the `Buffer` contains only a single value.
    /// By derefencing this `Buffer`, you obtain this value as well (which is probably preferred).
    ///
    /// # Example
    ///
    /// ```
    /// use custos::Buffer;
    ///
    /// let mut x: Buffer<f32, _> = 7f32.into();
    /// assert_eq!(**x, 7.);
    /// *x.item_mut() += 1.;
    /// assert_eq!(*x.item_mut(), 8.);
    ///
    /// ```   
    #[inline]
    pub fn item_mut(&mut self) -> &mut T
    where
        T: Unit,
    {
        &mut self.data.num
    }
}

impl<T> Deref for Num<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.num
    }
}

impl<T> DerefMut for Num<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.num
    }
}

#[cfg(test)]
mod tests {
    use crate::Buffer;

    #[test]
    fn test_deref() {
        let a = Buffer::from(5);
        let b = Buffer::from(7);

        let c = **a + **b;
        assert_eq!(c, 12);
    }

    #[test]
    fn test_deref_mut() {
        let mut a = Buffer::from(5);
        **a += 10;
        assert_eq!(**a, 15);
    }

    #[test]
    fn test_num_device() {
        use crate::Device;

        <()>::new().unwrap();
    }
}

use core::{
    convert::Infallible,
    ffi::c_void,
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use crate::{flag::AllocFlag, Alloc, Buffer, CloneBuf, CommonPtrs, Device, HasId, PtrType};

#[derive(Debug, Default)]
/// Makes it possible to use a single number in a [`Buffer`].
pub struct Num<T> {
    /// The stored number.
    pub num: T,
}

impl<T> PtrType for Num<T> {
    #[inline]
    fn size(&self) -> usize {
        0
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        crate::flag::AllocFlag::Num
    }
}

impl<T> CommonPtrs<T> for Num<T> {
    #[inline]
    fn ptrs(&self) -> (*const T, *mut c_void, u64) {
        (&self.num as *const T, null_mut(), 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64) {
        (&mut self.num as *mut T, null_mut(), 0)
    }
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

impl Device for () {
    type Data<T, S: crate::Shape> = Num<T>;
    type Error = Infallible;

    fn new() -> Result<Self, Infallible> {
        Ok(())
    }
}

impl<T: Default> Alloc<T> for () {
    #[inline]
    fn alloc<S: crate::Shape>(&self, _len: usize, _flag: AllocFlag) -> Self::Data<T, S> {
        Num::default()
    }

    #[inline]
    fn alloc_from_slice<S: crate::Shape>(
        &self,
        data: &[T],
        _alloc_flag: AllocFlag,
    ) -> Self::Data<T, S>
    where
        T: Clone,
    {
        data[0].clone().into()
    }
}

impl<'a, T: Clone> CloneBuf<'a, T> for () {
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

impl<'a, T> Buffer<'a, T, ()> {
    /// A [`Num`] [`Buffer`] is safe to copy.
    /// This method returns a new "[`Buffer`]" with the same single value.
    #[inline]
    pub fn copy(&self) -> Self
    where
        T: Copy,
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
    /// assert_eq!(*x, 7.);
    /// assert_eq!(x.item(), 7.);
    ///
    /// ```
    #[inline]
    pub fn item(&self) -> T
    where
        T: Copy,
    {
        self.data.num
    }
}

impl<'a, T> Deref for Buffer<'a, T, ()> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data.num
    }
}

impl<'a, T> DerefMut for Buffer<'a, T, ()> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data.num
    }
}

#[cfg(test)]
mod tests {
    use crate::Buffer;

    #[test]
    fn test_deref() {
        let a = Buffer::from(5);
        let b = Buffer::from(7);

        let c = *a + *b;
        assert_eq!(c, 12);
    }

    #[test]
    fn test_deref_mut() {
        let mut a = Buffer::from(5);
        *a += 10;
        assert_eq!(*a, 15);
    }

    #[test]
    fn test_num_device() {
        use crate::Device;

        let _device = <()>::new().unwrap();
    }
}

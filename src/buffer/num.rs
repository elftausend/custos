use core::{
    ffi::c_void,
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use crate::{shape::Shape, Buffer, CloneBuf, CommonPtrs, Device, Node, PtrType};

pub struct Num<T> {
    pub num: T,
}

impl<T> PtrType for Num<T> {
    #[inline]
    fn len(&self) -> usize {
        0
    }

    fn flag(&self) -> crate::flag::AllocFlag {
        crate::flag::AllocFlag::None
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

impl Device for () {
    type Ptr<U, S: Shape> = Num<U>;
    type Cache<'a> = ();

    fn new() -> crate::Result<Self> {
        Ok(())
    }
}

impl<'a, T: Clone> CloneBuf<'a, T> for () {
    #[inline]
    fn clone_buf(&self, buf: &Buffer<'a, T, Self>) -> Buffer<'a, T, Self> {
        Buffer {
            ptr: Num {
                num: buf.ptr.num.clone(),
            },
            device: buf.device,
            node: buf.node,
        }
    }
}

impl<T: crate::number::Number> From<T> for Buffer<'_, T, ()> {
    #[inline]
    fn from(ptr: T) -> Self {
        Buffer {
            ptr: Num { num: ptr },
            device: None,
            node: Node::default(),
        }
    }
}

impl<'a, T> Buffer<'a, T, ()> {
    #[inline]
    pub fn copy(&self) -> Self
    where
        T: Copy,
    {
        Buffer {
            ptr: Num { num: self.ptr.num },
            device: self.device,
            node: self.node,
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
        self.ptr.num
    }
}

impl<'a, T> Deref for Buffer<'a, T, ()> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.ptr.num
    }
}

impl<'a, T> DerefMut for Buffer<'a, T, ()> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr.num
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
}

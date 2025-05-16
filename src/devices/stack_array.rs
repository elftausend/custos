use core::ops::{Deref, DerefMut};

use crate::{HasId, HostPtr, PtrType, ShallowCopy, Unit, shape::Shape};

/// A possibly multi-dimensional array allocated on the stack.
/// It uses `S:`[`Shape`] to get the type of the array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackArray<S: Shape, T> {
    pub(crate) array: S::ARR<T>,
}

impl<S: Shape, T: Default + Copy> StackArray<S, T> {
    /// Creates a new `StackArray`.
    #[inline]
    pub fn new() -> Self {
        const {
            assert!(
                S::LEN > 0,
                "The size (N) of a stack allocated buffer must be greater than 0."
            )
        };
        StackArray { array: S::new() }
    }

    /// Returns a reference to the possibly multi-dimensional array.
    #[inline]
    pub const fn array(&self) -> &S::ARR<T> {
        &self.array
    }

    /// Returns a mutable reference to the possibly multi-dimensional array.
    #[inline]
    pub fn array_mut(&mut self) -> &mut S::ARR<T> {
        &mut self.array
    }
}

impl<T, S: Shape> HasId for StackArray<S, T> {
    #[inline]
    fn id(&self) -> crate::Id {
        crate::Id {
            id: self.as_ptr() as u64,
            len: self.len(),
        }
    }
}

impl<S: Shape, T: Default + Copy> Default for StackArray<S, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Shape, T> StackArray<S, T> {
    /// Creates a new `StackArray` from a possibly multi-dimensional array.
    pub fn from_array(array: S::ARR<T>) -> Self {
        const {
            assert!(
                S::LEN > 0,
                "The size (N) of a stack allocated buffer must be greater than 0."
            )
        };

        StackArray { array }
    }
}

impl<S: Shape, T> StackArray<S, T> {
    /// Returns a pointer to the possibly multi-dimensional array.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        &self.array as *const S::ARR<T> as *const T
    }

    /// Returns a pointer to the possibly multi-dimensional array.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> *mut T {
        &mut self.array as *mut S::ARR<T> as *mut T
    }

    /// Flattens a possibly multidimensional array.
    /// `&[[T], ..]` -> `&[T]`
    #[inline]
    pub const fn flatten(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), S::LEN) }
    }

    /// Flattens a possibly multidimensional array.
    /// `&mut [[T], ..]` -> `&mut [T]`
    #[inline]
    pub fn flatten_mut(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.as_ptr_mut(), S::LEN) }
    }
}

impl<S: Shape, T> Deref for StackArray<S, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.flatten()
    }
}

impl<S: Shape, T> DerefMut for StackArray<S, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.flatten_mut()
    }
}

impl<S: Shape, T: Unit> PtrType for StackArray<S, T> {
    #[inline]
    fn size(&self) -> usize {
        S::LEN
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        crate::flag::AllocFlag::None
    }

    #[inline]
    unsafe fn set_flag(&mut self, _flag: crate::flag::AllocFlag) {}
}

impl<S: Shape, T: Unit> HostPtr<T> for StackArray<S, T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.as_ptr()
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.as_ptr_mut()
    }
}

impl<S: Shape, T> ShallowCopy for StackArray<S, T>
where
    S::ARR<T>: Copy,
{
    #[inline]
    unsafe fn shallow(&self) -> Self {
        StackArray { array: self.array }
    }
}

#[cfg(test)]
mod test {
    // use crate::StackArray;

    // compile time error instead!
    // #[test]
    // #[should_panic]
    // fn test_stack_array_zero_len() {
    //     StackArray::<(), f32>::new();
    // }
}

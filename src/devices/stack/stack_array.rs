use core::{
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use crate::{shape::Shape, CommonPtrs, PtrType, ShallowCopy};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackArray<S: Shape, T> {
    pub array: S::ARR<T>,
    _private: (),
}

impl<S: Shape, T: Default + Copy> StackArray<S, T> {
    #[inline]
    pub fn new() -> Self {
        // TODO: one day... use const expressions
        assert!(S::LEN > 0,
            "The size (N) of a stack allocated buffer must be greater than 0."
        );
        StackArray { array: S::new(), _private: () }
    }
}

impl<S: Shape, T: Default + Copy> Default for StackArray<S, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Shape, T> StackArray<S, T> {
    pub fn from_array(array: S::ARR<T>) -> Self {
        assert!(S::LEN > 0,
            "The size (N) of a stack allocated buffer must be greater than 0."
        );
        
        StackArray {
            array,
            _private: ()
        }
    }
}

impl<S: Shape, T> StackArray<S, T> {
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        &self.array as *const S::ARR<T> as *const T
    }

    #[inline]
    pub fn as_ptr_mut(&mut self) -> *mut T {
        &mut self.array as *mut S::ARR<T> as *mut T
    }

    #[inline]
    pub const unsafe fn flatten(&self) -> &[T] {
        core::slice::from_raw_parts(self.as_ptr(), S::LEN)
    }

    #[inline]
    pub unsafe fn flatten_mut(&mut self) -> &mut [T] {
        core::slice::from_raw_parts_mut(self.as_ptr_mut(), S::LEN)
    }
}

impl<S: Shape, T> Deref for StackArray<S, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.flatten() }
    }
}

impl<S: Shape, T> DerefMut for StackArray<S, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.flatten_mut() }
    }
}

impl<S: Shape, T> PtrType for StackArray<S, T> {
    #[inline]
    fn len(&self) -> usize {
        S::LEN
    }

    #[inline]
    fn flag(&self) -> crate::flag::AllocFlag {
        crate::flag::AllocFlag::None
    }
}

impl<S: Shape, T> CommonPtrs<T> for StackArray<S, T> {
    #[inline]
    fn ptrs(&self) -> (*const T, *mut core::ffi::c_void, u64) {
        (self.as_ptr(), null_mut(), 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut core::ffi::c_void, u64) {
        (self.as_mut_ptr(), null_mut(), 0)
    }
}

impl<S: Shape, T> ShallowCopy for StackArray<S, T>
where
    S::ARR<T>: Copy,
{
    #[inline]
    unsafe fn shallow(&self) -> Self {
        StackArray { array: self.array, _private: () }
    }
}

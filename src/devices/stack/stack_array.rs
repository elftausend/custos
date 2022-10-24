use core::{ops::{Deref, DerefMut}, ptr::null_mut};

use crate::PtrType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackArray<const N: usize, T = f32> {
    pub array: [T; N],
}

impl<T, const N: usize> StackArray<N, T> {
    #[inline]
    pub fn new(array: [T; N]) -> Self {
        StackArray { array }
    }
}

impl<const N: usize, T> Deref for StackArray<N, T> {
    type Target = [T; N];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl<const N: usize, T> DerefMut for StackArray<N, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.array
    }
}

impl<const N: usize, T> PtrType<T> for StackArray<N, T> {
    unsafe fn dealloc(&mut self, _len: usize) {}

    #[inline]
    fn ptrs(&self) -> (*const T, *mut core::ffi::c_void, u64) {
        (self.array.as_ptr(), null_mut(), 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut core::ffi::c_void, u64) {
        (self.array.as_mut_ptr(), null_mut(), 0)
    }

    fn from_ptrs(_ptrs: (*mut T, *mut core::ffi::c_void, u64)) -> Self {
        unimplemented!("Cannot create a StackArray from pointers.");
    }
}
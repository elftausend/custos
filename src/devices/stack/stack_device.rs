use std::{
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use crate::{devices::CacheAble, Alloc, Buffer, Device, PtrType, CPUCL, IsCPU, DevicelessAble};

#[derive(Debug, Clone, Copy)]
pub struct Stack;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackArray<const N: usize, T = f32> {
    pub array: [T; N],
}

impl<T, const N: usize> StackArray<N, T> {
    #[inline]
    pub fn new(array: [T; N]) -> Self {
        StackArray {
            array
        }
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
    fn ptrs(&self) -> (*const T, *mut std::ffi::c_void, u64) {
        (self.array.as_ptr(), null_mut(), 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut std::ffi::c_void, u64) {
        (self.array.as_mut_ptr(), null_mut(), 0)
    }

    fn from_ptrs(_ptrs: (*mut T, *mut std::ffi::c_void, u64)) -> Self {
        unimplemented!("Cannot create a StackArray from pointers.");
    }
}

pub struct StackRetrieve {}

impl<const N: usize> CacheAble<Stack, N> for StackRetrieve {
    fn retrieve<T>(
        device: &Stack,
        len: usize,
        _add_node: impl crate::AddGraph,
    ) -> crate::Buffer<T, Stack, N>
    where
        Stack: Alloc<T, N>,
    {
        Buffer::new(device, len)
    }
}

impl<T: Copy + Default> DevicelessAble<T> for Stack {}

impl Device for Stack {
    type Ptr<U, const N: usize> = StackArray<N, U>;
    type Cache<const N: usize> = StackRetrieve;
}

impl IsCPU for Stack {}
impl CPUCL for Stack {}

impl<const N: usize, T: Copy + Default> Alloc<T, N> for Stack {
    #[inline]
    fn alloc(&self, _len: usize) -> StackArray<N, T> {
        // TODO: one day... use const expressions
        if N == 0 {
            panic!("The size (N) of a stack allocated buffer must be greater than 0.");
        }
        StackArray {
            array: [T::default(); N],
        }
    }

    #[inline]
    fn with_slice(&self, data: &[T]) -> StackArray<N, T> {
        let mut array = self.alloc(0);
        array.copy_from_slice(&data[..N]);
        array
    }

    #[inline]
    fn with_array(&self, array: [T; N]) -> <Self as Device>::Ptr<T, N> {
        StackArray { array }
    }
}

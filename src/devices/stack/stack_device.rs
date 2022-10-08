use std::{ptr::null_mut, ops::{Deref, DerefMut}};

use crate::{Device, PtrType, Alloc, CPUCL, devices::CacheAble};

#[derive(Debug, Clone, Copy)]
pub struct Stack;

#[derive(Debug, Clone, Copy)]
pub struct StackArray<const N: usize, T = f32> {
    array: [T; N]
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
    unsafe fn dealloc(&mut self, _len: usize) {
        return;
    }

    fn ptrs(&self) -> (*mut T, *mut std::ffi::c_void, u64) {
        (self.array.as_ptr() as *mut T, null_mut(), 0)
    }

    fn from_ptrs(_ptrs: (*mut T, *mut std::ffi::c_void, u64)) -> Self {
        unimplemented!("Cannot create a StackArray from pointers.");
    }
}

pub struct StackRetrieve {
}

impl<const N: usize> CacheAble<Stack, N> for StackRetrieve {
    fn retrieve<'a, T>(device: &'a Stack, len: usize, add_node: impl crate::AddGraph) -> crate::Buffer<'a, T, Stack, N>
    where Stack: Alloc<T, N> 
    {
        todo!()
    }
}
impl Device for Stack {
    type Ptr<U, const N: usize> = StackArray<N, U>;
    type Cache<const N: usize> = StackRetrieve;
}

impl CPUCL for Stack {}

impl<const N: usize, T: Copy + Default> Alloc<T, N> for Stack {
    #[inline]
    fn alloc(&self, _len: usize) -> StackArray<N, T> {
        StackArray { array: [T::default(); N] }
    }

    fn from_slice(&self, data: &[T]) -> StackArray<N, T> {
        let mut array = self.alloc(0);
        array.copy_from_slice(&data[..N]);
        array
    }

    #[inline]
    fn from_array(&self, array: [T; N]) -> <Self as Device>::Ptr<T, N> {
        StackArray {
            array
        }
    }
}
use core::{marker::PhantomData, ptr::null_mut};

use crate::{flag::AllocFlag, CommonPtrs, HasId, Id, PtrType, ShallowCopy};

use super::api::cufree;

/// The pointer used for `CUDA` [`Buffer`](crate::Buffer)s
#[derive(Debug, PartialEq, Eq)]
pub struct CUDAPtr<T> {
    /// The pointer to the CUDA memory object.
    pub ptr: u64,
    /// The number of elements addressable
    pub len: usize,
    /// Allocation flag for the pointer.
    pub flag: AllocFlag,
    pub p: PhantomData<T>,
}

impl<T> HasId for CUDAPtr<T> {
    #[inline]
    fn id(&self) -> Id {
        Id {
            id: self.ptr,
            len: self.len,
        }
    }

    #[inline]
    unsafe fn set_id(&mut self, id: u64) {
        self.ptr = id
    }
}

impl<T> Default for CUDAPtr<T> {
    #[inline]
    fn default() -> Self {
        Self {
            ptr: 0,
            len: 0,
            flag: AllocFlag::default(),
            p: PhantomData,
        }
    }
}

impl<T> Drop for CUDAPtr<T> {
    fn drop(&mut self) {
        if !matches!(self.flag, AllocFlag::None | AllocFlag::BorrowedCache) {
            return;
        }

        if self.ptr == 0 {
            return;
        }
        unsafe {
            cufree(self.ptr).unwrap();
        }
    }
}

impl<T> ShallowCopy for CUDAPtr<T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        CUDAPtr {
            ptr: self.ptr,
            len: self.len,
            flag: AllocFlag::Wrapper,
            p: PhantomData,
        }
    }
}

impl<T> PtrType for CUDAPtr<T> {
    #[inline]
    fn size(&self) -> usize {
        self.len
    }

    #[inline]
    unsafe fn set_size(&mut self, size: usize) {
        self.len = size
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.flag
    }
}

impl<T> CommonPtrs<T> for CUDAPtr<T> {
    #[inline]
    fn ptrs(&self) -> (*const T, *mut std::ffi::c_void, u64) {
        (null_mut(), null_mut(), self.ptr)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut std::ffi::c_void, u64) {
        (null_mut(), null_mut(), self.ptr)
    }
}

use core::{ffi::c_void, ptr::null_mut};

#[cfg(unified_cl)]
use core::ops::{Deref, DerefMut};

#[cfg(unified_cl)]
use crate::HostPtr;

use min_cl::api::release_mem_object;

use crate::{flag::AllocFlag, HasId, Id, PtrType, ShallowCopy};

/// The pointer used for `OpenCL` [`Buffer`](crate::Buffer)s
#[derive(Debug, PartialEq, Eq)]
pub struct CLPtr<T> {
    /// The pointer to the OpenCL memory object
    pub ptr: *mut c_void,
    /// Possibly a pointer to the host memory. Only active for devices with unified memory.
    pub host_ptr: *mut T,
    /// The number of elements allocated
    pub len: usize,
    /// The flag of the memory object
    pub flag: AllocFlag,
}

unsafe impl<T: Sync> Sync for CLPtr<T> {}
unsafe impl<T: Send> Send for CLPtr<T> {}

impl<T> Default for CLPtr<T> {
    #[inline]
    fn default() -> Self {
        Self {
            ptr: null_mut(),
            host_ptr: null_mut(),
            len: 0,
            flag: AllocFlag::default(),
        }
    }
}

impl<T> HasId for CLPtr<T> {
    #[inline]
    fn id(&self) -> Id {
        Id {
            id: self.ptr as u64,
            len: self.len,
        }
    }
}

impl<T> CLPtr<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> ShallowCopy for CLPtr<T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        CLPtr {
            ptr: self.ptr,
            host_ptr: self.host_ptr,
            len: self.len,
            flag: AllocFlag::Wrapper,
        }
    }
}

impl<T> PtrType for CLPtr<T> {
    #[inline]
    fn size(&self) -> usize {
        self.len
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.flag
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: AllocFlag) {
        self.flag = flag;
    }
}

#[cfg(unified_cl)]
impl<T> HostPtr<T> for CLPtr<T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.host_ptr
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.host_ptr
    }
}

#[cfg(unified_cl)]
impl<T> Deref for CLPtr<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.as_slice() }
    }
}

#[cfg(unified_cl)]
impl<T> DerefMut for CLPtr<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.as_mut_slice() }
    }
}

impl<T> Drop for CLPtr<T> {
    fn drop(&mut self) {
        if !self.flag.continue_deallocation() {
            return;
        }

        if self.ptr.is_null() {
            return;
        }
        unsafe {
            release_mem_object(self.ptr).unwrap();
        }
    }
}

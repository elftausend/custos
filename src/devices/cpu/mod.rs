//! The CPU module provides the CPU backend for custos.

use crate::{
    module_comb::{HasId, Id},
    CommonPtrs, PtrType, ShallowCopy,
};
#[cfg(feature = "blas")]
pub use blas::*;
use core::{
    alloc::Layout,
    mem::{align_of, size_of},
    ptr::null_mut,
};
pub use cpu_device::*;
use std::alloc::handle_alloc_error;

use crate::flag::AllocFlag;

#[cfg(feature = "blas")]
mod blas;
mod cpu_device;
mod ops;

/// The pointer used for `CPU` [`Buffer`](crate::Buffer)s
#[derive(PartialEq, Eq, Debug)]
pub struct CPUPtr<T> {
    /// The pointer to the data
    pub ptr: *mut T,
    /// The length of the data
    pub len: usize,
    /// Allocation flag for the pointer
    pub flag: AllocFlag,
    /// The alignment of type `T`
    pub align: Option<usize>,
    /// The size of type `T`
    pub size: Option<usize>,
}

impl<T> CPUPtr<T> {
    /// Create a new `CPUPtr` with the given length and allocation flag
    ///
    /// # Safety
    ///
    /// The allocated memory is not initialized.
    /// Make sure that the memory was written to before being read.
    ///
    /// # Example
    /// ```
    /// use custos::{cpu::CPUPtr, flag::AllocFlag};
    ///
    /// let ptr = unsafe { CPUPtr::<f32>::new(10, AllocFlag::None) };
    /// assert_eq!(ptr.len, 10);
    /// assert_eq!(ptr.flag, AllocFlag::None);
    /// assert_eq!(ptr.ptr.is_null(), false);
    /// ```
    pub unsafe fn new(len: usize, flag: AllocFlag) -> CPUPtr<T> {
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        CPUPtr::from_ptr(ptr.cast(), len, flag)
    }

    /// Create a new `CPUPtr` with the given length and allocation flag. Initializes memory as well.
    /// # Example
    /// ```
    /// use custos::{cpu::CPUPtr, flag::AllocFlag};
    ///
    /// let ptr = CPUPtr::<f32>::new_initialized(10, AllocFlag::None);
    /// assert_eq!(ptr.len, 10);
    /// assert_eq!(ptr.flag, AllocFlag::None);
    /// assert_eq!(ptr.ptr.is_null(), false);
    /// ```
    pub fn new_initialized(len: usize, flag: AllocFlag) -> CPUPtr<T> {
        let cpu_ptr = unsafe { CPUPtr::new(len, flag) };

        // initialize block of memory
        for element in
            unsafe { std::slice::from_raw_parts_mut(cpu_ptr.ptr as *mut u8, len * size_of::<T>()) }
        {
            *element = 0;
        }

        cpu_ptr
    }

    /// Wrap a raw pointer with the given length and allocation flag into a `CPUPtr`
    /// Depending on the flag: [`AllocFlag`], the pointer will be freed or left untouched when the `CPUPtr` is dropped.
    ///
    /// # Safety
    /// Basically the same as for [Vec]::from_raw_parts.
    ///
    /// # Example
    /// ```
    /// use custos::{cpu::CPUPtr, flag::AllocFlag};
    ///
    /// let ptr = CPUPtr::<f32>::new_initialized(10, AllocFlag::None);
    /// // AllocFlag::Wrapper will not free the pointer -> prevents double free
    /// let ptr2 = unsafe { CPUPtr::<f32>::from_ptr(ptr.ptr, 10, AllocFlag::Wrapper) };
    /// assert_eq!(ptr.ptr, ptr2.ptr);
    /// ```
    #[inline]
    pub unsafe fn from_ptr(ptr: *mut T, len: usize, flag: AllocFlag) -> CPUPtr<T> {
        CPUPtr {
            ptr,
            len,
            flag,
            align: None,
            size: None,
        }
    }
}

impl<T> HasId for CPUPtr<T> {
    #[inline]
    fn id(&self) -> Id {
        Id {
            id: self.ptr as u64,
            len: self.len,
        }
    }
}

impl<T> Default for CPUPtr<T> {
    fn default() -> Self {
        Self {
            ptr: null_mut(),
            flag: AllocFlag::default(),
            len: 0,
            align: None,
            size: None,
        }
    }
}

impl<T> Drop for CPUPtr<T> {
    fn drop(&mut self) {
        if !matches!(self.flag, AllocFlag::None | AllocFlag::BorrowedCache) {
            return;
        }

        if self.ptr.is_null() {
            return;
        }

        let (align, size) = if let Some(align) = self.align {
            (align, self.size.expect("size must be set if align is set"))
        } else {
            (align_of::<T>(), size_of::<T>())
        };

        let layout = Layout::from_size_align(self.len * size, align).unwrap();

        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, layout);
        }
    }
}

impl<T> PtrType for CPUPtr<T> {
    #[inline]
    fn size(&self) -> usize {
        self.len
    }

    #[inline]
    fn flag(&self) -> AllocFlag {
        self.flag
    }
}

impl<T> CommonPtrs<T> for CPUPtr<T> {
    #[inline]
    fn ptrs(&self) -> (*const T, *mut core::ffi::c_void, u64) {
        (self.ptr as *const T, null_mut(), 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut core::ffi::c_void, u64) {
        (self.ptr as *mut T, null_mut(), 0)
    }
}

impl<T> ShallowCopy for CPUPtr<T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        CPUPtr {
            ptr: self.ptr,
            len: self.len,
            flag: AllocFlag::Wrapper,
            align: self.align,
            size: self.size,
        }
    }
}

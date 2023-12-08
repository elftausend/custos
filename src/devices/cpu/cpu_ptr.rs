use core::{
    alloc::Layout,
    mem::{align_of, size_of},
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use std::alloc::handle_alloc_error;

use crate::{flag::AllocFlag, CommonPtrs, HasId, Id, PtrType, ShallowCopy};

/// The pointer used for `CPU` [`Buffer`](crate::Buffer)s
#[derive(Debug)]
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

impl<T: PartialEq> PartialEq for CPUPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for CPUPtr<T> {}

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

    /// Extracts a slice containing the entire `CPUPtr`.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Extracts a mutable slice of the entire `CPUPtr`.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Returns the layout info of the `CPUPtr`
    /// If the `align` and `size` field are set,
    /// they will be used, otherwise the size and alignment are determined by the type `T`
    #[inline]
    pub fn layout_info(&self) -> (usize, usize) {
        let (align, size) = if let Some(align) = self.align {
            (align, self.size.expect("size must be set if align is set"))
        } else {
            (align_of::<T>(), size_of::<T>())
        };

        (align, size)
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
impl<T> Deref for CPUPtr<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for CPUPtr<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
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

    #[inline]
    unsafe fn set_flag(&mut self, flag: AllocFlag) {
        self.flag = flag
    }
}

impl<T> CommonPtrs<T> for CPUPtr<T> {
    #[inline]
    fn ptrs(&self) -> (*const T, *mut core::ffi::c_void, u64) {
        (self.ptr as *const T, null_mut(), 0)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut core::ffi::c_void, u64) {
        (self.ptr, null_mut(), 0)
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

#[cfg(feature = "serde")]
pub mod serde {
    use core::{fmt, marker::PhantomData};

    use serde::{
        de::{SeqAccess, Visitor},
        ser::SerializeSeq,
        Deserialize,
    };

    use super::CPUPtr;

    impl<T: serde::Serialize> serde::Serialize for CPUPtr<T> {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut seq = serializer.serialize_seq(Some(self.len()))?;

            for e in self.iter() {
                seq.serialize_element(e)?;
            }

            seq.end()
        }
    }
    pub struct CpuPtrVisitor<T> {
        pub marker: PhantomData<T>,
    }

    impl<'de, T> Visitor<'de> for CpuPtrVisitor<T>
    where
        T: Deserialize<'de>,
    {
        type Value = CPUPtr<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let values = unsafe {
                CPUPtr::<T>::new(
                    seq.size_hint().unwrap_or_default(),
                    crate::flag::AllocFlag::None,
                )
            };

            let mut offset = 0;
            while let Some(value) = seq.next_element::<T>()? {
                unsafe {
                    let end = values.ptr.add(offset);
                    end.write(value)
                }
                offset += 1;
            }

            Ok(values)
        }
    }

    impl<'a, T: serde::Deserialize<'a>> serde::Deserialize<'a> for CPUPtr<T> {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'a>,
        {
            deserializer.deserialize_seq(CpuPtrVisitor {
                marker: PhantomData,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use serde_test::{assert_tokens, Token};

        use crate::cpu::CPUPtr;

        #[test]
        fn test_ser_de_of_cpu_ptr_filled() {
            let mut cpu_ptr = CPUPtr::<i32>::new_initialized(10, crate::flag::AllocFlag::None);
            cpu_ptr
                .as_mut_slice()
                .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            assert_tokens(
                &cpu_ptr,
                &[
                    Token::Seq { len: Some(10) },
                    Token::I32(1),
                    Token::I32(2),
                    Token::I32(3),
                    Token::I32(4),
                    Token::I32(5),
                    Token::I32(6),
                    Token::I32(7),
                    Token::I32(8),
                    Token::I32(9),
                    Token::I32(10),
                    Token::SeqEnd,
                ],
            );
        }
    }
}

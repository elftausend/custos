use core::{
    alloc::Layout,
    mem::{align_of, size_of},
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use std::alloc::handle_alloc_error;

use crate::{flag::AllocFlag, HasId, HostPtr, Id, PtrType, ShallowCopy};

/// The pointer used for `CPU` [`Buffer`](crate::Buffer)s
#[derive(Debug)]
pub struct CPUPtr<T> {
    /// The pointer to the data
    pub ptr: *mut T,
    /// The length of the data
    pub len: usize,
    /// Allocation flag for the pointer
    pub flag: AllocFlag,
}

unsafe impl<T: Send> Send for CPUPtr<T> {}
unsafe impl<T: Sync> Sync for CPUPtr<T> {}

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
        CPUPtr { ptr, len, flag }
    }
    pub fn from_vec(mut vec: Vec<T>) -> CPUPtr<T> {
        // CPUPtr only knows about the length, not the capacity -> deallocation happens with length, which may be less than the capacity
        vec.shrink_to_fit();

        let mut ptr = vec.as_mut_ptr();

        // Vec uses dangling invalid pointer when capacity is 0 (instead of just null)
        // -> would deallocate this dangling pointer on drop
        if vec.capacity() == 0 {
            ptr = std::ptr::null_mut()
        }

        let len = vec.len();
        core::mem::forget(vec);

        unsafe { CPUPtr::from_ptr(ptr, len, AllocFlag::None) }
    }

    // pub fn

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
        (align_of::<T>(), size_of::<T>())
    }

    pub fn current_memory(&self) -> Option<(*mut u8, Layout)> {
        if self.ptr.is_null() || size_of::<T>() == 0 {
            return None;
        }
        let (align, size) = self.layout_info();
        let layout = Layout::from_size_align(self.len * size, align).ok()?;
        Some((self.ptr.cast(), layout))
    }
}

impl<T> HostPtr<T> for CPUPtr<T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.ptr
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
        }
    }
}

impl<T> Drop for CPUPtr<T> {
    fn drop(&mut self) {
        if !self.flag.continue_deallocation() {
            return;
        }

        if let Some((ptr, layout)) = self.current_memory() {
            unsafe {
                std::alloc::dealloc(ptr, layout);
            }
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

impl<T> ShallowCopy for CPUPtr<T> {
    #[inline]
    unsafe fn shallow(&self) -> Self {
        CPUPtr {
            ptr: self.ptr,
            len: self.len,
            flag: AllocFlag::Wrapper,
        }
    }
}

pub struct DeallocWithLayout {
    ptr: core::mem::ManuallyDrop<CPUPtr<u8>>,
    layout: Layout,
}

impl DeallocWithLayout {
    #[inline]
    pub unsafe fn new<T>(ptr: CPUPtr<T>) -> Option<Self> {
        let (_, layout) = ptr.current_memory()?;
        let ptr = core::mem::ManuallyDrop::new(ptr);
        Some(Self {
            ptr: core::mem::ManuallyDrop::new(CPUPtr {
                ptr: ptr.ptr as *mut u8,
                len: ptr.len,
                flag: ptr.flag,
            }),
            layout,
        })
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl core::ops::Deref for DeallocWithLayout {
    type Target = CPUPtr<u8>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl core::ops::DerefMut for DeallocWithLayout {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}

impl Drop for DeallocWithLayout {
    fn drop(&mut self) {
        if !self.ptr.flag.continue_deallocation() {
            return;
        }

        unsafe {
            std::alloc::dealloc(self.ptr.ptr, self.layout);
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

#[cfg(test)]
mod tests {
    use core::{alloc::Layout, marker::PhantomData};

    use super::{CPUPtr, DeallocWithLayout};

    #[test]
    fn test_return_current_memory() {
        let data = CPUPtr::<f32>::new_initialized(10, crate::flag::AllocFlag::None);

        let ret = data.current_memory().unwrap();
        assert_eq!(ret.0, data.ptr as *mut u8);
        assert_eq!(ret.1, Layout::from_size_align(data.len * 4, 4).unwrap());
    }

    #[test]
    fn test_alloc_from_empty_vec() {
        let data = Vec::<f32>::new();
        let res = CPUPtr::from_vec(data);
        assert!(res.ptr.is_null());
        assert_eq!(res.len, 0);
    }

    #[test]
    fn test_alloc_from_excess_cap_vec() {
        let mut data = Vec::<f32>::new();
        for _ in 0..10 {
            data.push(4.);
        }
        let res = CPUPtr::from_vec(data);
        assert!(!res.ptr.is_null());
        assert_eq!(res.len, 10);
    }

    #[test]
    fn test_alloc_from_vec_with_zst() {
        let mut data = Vec::<PhantomData<f32>>::new();
        for _ in 0..10 {
            data.push(PhantomData);
        }
        let res = CPUPtr::from_vec(data);
        assert!(!res.ptr.is_null());
        assert_eq!(res.len, 10);
    }

    #[test]
    fn test_dealloc_with_layout() {
        let data = CPUPtr::<f32>::new_initialized(10, crate::flag::AllocFlag::None);
        let dealloc = unsafe { DeallocWithLayout::new(data).unwrap() };
        assert_eq!(dealloc.layout().size(), 40)
    }
}

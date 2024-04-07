use core::{marker::PhantomData, ptr::null_mut};

use crate::{flag::AllocFlag, CommonPtrs, HasId, Id, PtrType, ShallowCopy};

use super::api::{cu_read, cufree, cumalloc};

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

impl<T> CUDAPtr<T> {
    pub fn new(len: usize, flag: AllocFlag) -> Self {
        let ptr = cumalloc::<T>(len).unwrap();
        // TODO: use unified mem if available -> i can't test this
        CUDAPtr {
            ptr,
            len,
            flag,
            p: PhantomData,
        }
    }

    pub fn read(&self) -> Vec<T>
    where
        T: Default + Clone,
    {
        let mut data = vec![T::default(); self.len];
        cu_read(&mut data, self.ptr).unwrap();
        data
    }
}

impl<T> HasId for CUDAPtr<T> {
    #[inline]
    fn id(&self) -> Id {
        Id {
            id: self.ptr,
            len: self.len,
        }
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
        if !self.flag.continue_deallocation() {
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
    fn flag(&self) -> AllocFlag {
        self.flag
    }

    #[inline]
    unsafe fn set_flag(&mut self, flag: AllocFlag) {
        self.flag = flag;
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

#[cfg(feature = "serde")]
mod serde {
    use super::CUDAPtr;
    use crate::{
        cpu::CPUPtr,
        cuda::api::{cu_read, cu_write},
    };
    use core::marker::PhantomData;

    impl<T: serde::Serialize> serde::Serialize for CUDAPtr<T> {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut data = unsafe { CPUPtr::<T>::new(self.len, crate::flag::AllocFlag::None) };
            cu_read(&mut data, self.ptr).unwrap();
            data.serialize(serializer)
        }
    }

    impl<'a, T: serde::Deserialize<'a>> serde::Deserialize<'a> for CUDAPtr<T> {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'a>,
        {
            let cpu_ptr = deserializer.deserialize_seq(crate::cpu::serde::CpuPtrVisitor {
                marker: PhantomData::<T>,
            })?;

            let cuda_ptr = CUDAPtr::new(cpu_ptr.len, crate::flag::AllocFlag::None);
            cu_write(cuda_ptr.ptr, &cpu_ptr).unwrap();
            Ok(cuda_ptr)
        }
    }

    #[cfg(test)]
    mod tests {
        use serde_test::{assert_ser_tokens, Token};

        use crate::{
            cuda::{api::cu_write, CUDAPtr},
            Base, CUDA,
        };

        #[test]
        fn test_ser_de_of_cuda_ptr_filled() {
            let _device = CUDA::<Base>::new(0).unwrap();

            let cuda_ptr = CUDAPtr::<i32>::new(10, crate::flag::AllocFlag::None);
            cu_write(cuda_ptr.ptr, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();
            let tokens = [
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
            ];
            assert_ser_tokens(&cuda_ptr, &tokens);

            //     let mut de = serde_test::de::Deserializer::new(&tokens);
            //     let mut deserialized_val = match T::deserialize(&mut de) {
            //         Ok(v) => {
            //             assert_eq!(v, *value);
            //             v
            //         }
            //         Err(e) => panic!("tokens failed to deserialize: {}", e),
            //     };
            //     if de.remaining() > 0 {
            //         panic!("{} remaining tokens", de.remaining());
            //     }
        }
    }
}

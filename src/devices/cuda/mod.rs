//! The CUDA module provides the CUDA backend for custos.

pub mod api;

/// Type alias for `core::ffi::c_ulonglong`. Used for CUDA memory object pointers.
pub type CUdeviceptr = core::ffi::c_ulonglong;

mod cuda_device;
mod kernel_cache;
mod kernel_launch;
mod ops;
mod source;

pub use source::*;

use std::{marker::PhantomData, ptr::null_mut};

pub use cuda_device::*;
pub use kernel_cache::*;
pub use kernel_launch::*;

use crate::{
    flag::AllocFlag,
    module_comb::{HasId, Id},
    Buffer, CDatatype, CommonPtrs, PtrType, ShallowCopy,
};

use self::api::cufree;

/// Another shorter type for Buffer<'a, T, CUDA, S>
pub type CUBuffer<'a, T> = Buffer<'a, T, CUDA>;

/// Reads the environment variable `CUSTOS_CU_DEVICE_IDX` and returns the value as a `usize`.
pub fn chosen_cu_idx() -> usize {
    std::env::var("CUSTOS_CU_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect(
            "Environment variable 'CUSTOS_CU_DEVICE_IDX' contains an invalid CUDA device index!",
        )
}

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
            id: self.ptr as u64,
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

/// Sets the elements of a CUDA Buffer to zero.
/// # Example
/// ```
/// use custos::{CUDA, Buffer, Read, cuda::cu_clear};
///
/// fn main() -> Result<(), custos::Error> {
///     let device = CUDA::new(0)?;
///     let mut lhs = Buffer::<i32, _>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(device.read(&lhs), vec![15, 30, 21, 5, 8]);
///
///     cu_clear(&device, &mut lhs);
///     assert_eq!(device.read(&lhs), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn cu_clear<T: CDatatype>(device: &CUDA, buf: &mut Buffer<T, CUDA>) -> crate::Result<()> {
    let src = format!(
        r#"extern "C" __global__ void clear({datatype}* self, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    self[idx] = 0;
                }}
                
            }}
    "#,
        datatype = T::as_c_type_str()
    );
    launch_kernel1d(buf.len(), device, &src, "clear", &[buf, &buf.len()])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::ffi::c_void;

    use crate::{
        cuda::{api::culaunch_kernel, fn_cache},
        Buffer, Read, CUDA,
    };

    #[test]
    fn test_cached_kernel_launch() -> crate::Result<()> {
        let device = CUDA::new(0)?;

        let a = Buffer::from((&device, [1, 2, 3, 4, 5]));
        let b = Buffer::from((&device, [4, 1, 7, 6, 9]));

        let c = Buffer::<i32, _>::new(&device, a.len());

        let src = r#"
            extern "C" __global__ void add(int *a, int *b, int *c, int numElements)
            {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {
                    c[idx] = a[idx] + b[idx];
                }
        }"#;

        for _ in 0..1000 {
            fn_cache(&device, src, "add")?;

            assert_eq!(device.kernel_cache.borrow().kernels.len(), 1);
        }
        let function = fn_cache(&device, src, "add")?;

        culaunch_kernel(
            &function,
            [a.len() as u32, 1, 1],
            [1, 1, 1],
            0,
            &mut device.stream(),
            &mut [
                &a.ptrs().2 as *const u64 as *mut c_void,
                &b.ptrs().2 as *const u64 as *mut c_void,
                &mut c.ptrs().2 as *mut u64 as *mut c_void,
                &a.len() as *const usize as *mut c_void,
            ],
        )?;

        assert_eq!(&vec![5, 3, 10, 10, 14], &device.read(&c));
        Ok(())
    }
}

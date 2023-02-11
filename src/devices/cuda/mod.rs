pub mod api;
mod cuda_device;
mod kernel_cache;
mod kernel_launch;
mod ops;

use std::{marker::PhantomData, ptr::null_mut};

pub use cuda_device::*;
pub use kernel_cache::*;
pub use kernel_launch::*;

use crate::{flag::AllocFlag, Buffer, CDatatype, CommonPtrs, PtrType, ShallowCopy};

use self::api::cufree;

pub type CUBuffer<'a, T> = Buffer<'a, T, CUDA>;

pub fn chosen_cu_idx() -> usize {
    std::env::var("CUSTOS_CU_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect(
            "Environment variable 'CUSTOS_CU_DEVICE_IDX' contains an invalid CUDA device index!",
        )
}

#[derive(Debug, PartialEq, Eq)]
pub struct CUDAPtr<T> {
    pub ptr: u64,
    pub len: usize,
    pub flag: AllocFlag,
    p: PhantomData<T>,
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
        if self.flag != AllocFlag::None {
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
    fn len(&self) -> usize {
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

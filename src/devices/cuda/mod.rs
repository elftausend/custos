pub mod api;
mod cuda_device;
mod kernel_cache;
mod kernel_launch;

use std::{marker::PhantomData, ptr::null_mut};

pub use cuda_device::*;
pub use kernel_cache::*;
pub use kernel_launch::*;

use crate::{Buffer, CDatatype, PtrType};

use self::api::cufree;

pub type CUBuffer<'a, T> = Buffer<'a, T, CUDA>;

pub fn chosen_cu_idx() -> usize {
    std::env::var("CUSTOS_CU_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect("Environment variable 'CUSTOS_CU_DEVICE_IDX' contains an invalid CUDA device index!")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CUDAPtr<T> {
    pub ptr: u64,
    p: PhantomData<T>,
}

impl<T> Default for CUDAPtr<T> {
    fn default() -> Self {
        Self {
            ptr: 0,
            p: PhantomData,
        }
    }
}

impl<T> PtrType<T> for CUDAPtr<T> {
    #[inline]
    unsafe fn dealloc(&mut self, _len: usize) {
        if self.ptr == 0 {
            return;
        }
        cufree(self.ptr).unwrap();
    }

    #[inline]
    fn ptrs(&self) -> (*const T, *mut std::ffi::c_void, u64) {
        (null_mut(), null_mut(), self.ptr)
    }

    #[inline]
    fn ptrs_mut(&mut self) -> (*mut T, *mut std::ffi::c_void, u64) {
        (null_mut(), null_mut(), self.ptr)
    }

    #[inline]
    unsafe fn from_ptrs(ptrs: (*mut T, *mut std::ffi::c_void, u64)) -> Self {
        Self {
            ptr: ptrs.2,
            p: PhantomData,
        }
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
    launch_kernel1d(buf.len, device, &src, "clear", &[buf, &buf.len])?;
    Ok(())
}

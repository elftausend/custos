//! The CUDA module provides the CUDA backend for custos.

pub mod api;

/// Type alias for `core::ffi::c_ulonglong`. Used for CUDA memory object pointers.
pub type CUdeviceptr = core::ffi::c_ulonglong;

mod cuda;
mod cuda_device;
mod cuda_ptr;
mod fusing;
mod kernel_cache;
mod kernel_launch;
#[cfg(feature = "lazy")]
pub mod lazy;
mod ops;
mod source;

pub use cuda_ptr::*;

pub use source::*;

pub use cuda::*;
pub use cuda_device::*;
pub use kernel_cache::*;
pub use kernel_launch::*;
pub use ops::*;

use crate::CDatatype;

/// Another type for [`CUDAPtr`]
pub type CUBuffer<T> = CUDAPtr<T>;

/// Reads the environment variable `CUSTOS_CU_DEVICE_IDX` and returns the value as a `usize`.
pub fn chosen_cu_idx() -> usize {
    std::env::var("CUSTOS_CU_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect(
            "Environment variable 'CUSTOS_CU_DEVICE_IDX' contains an invalid CUDA device index!",
        )
}

/// Sets the elements of a CUDA Buffer to zero.
/// # Example
/// ```
/// use custos::{CUDA, Buffer, Read, cuda::cu_clear, Base};
///
/// fn main() -> Result<(), custos::Error> {
///     let device = CUDA::<Base>::new(0)?;
///     let mut lhs = Buffer::<i32, _>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(lhs.read(), vec![15, 30, 21, 5, 8]);
///
///     cu_clear(&device, &mut lhs);
///     assert_eq!(lhs.read(), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn cu_clear<T: CDatatype>(device: &CudaDevice, buf: &mut CUDAPtr<T>) -> crate::Result<()> {
    let src = format!(
        r#"extern "C" __global__ void clear({datatype}* self, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    self[idx] = 0;
                }}
                
            }}
    "#,
        datatype = T::C_DTYPE_STR
    );
    device.launch_kernel1d(buf.len, &src, "clear", &[buf, &buf.len])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::ffi::c_void;

    use crate::{
        Base, Buffer, CUDA,
        cuda::{api::culaunch_kernel, fn_cache},
    };

    #[test]
    fn test_cached_kernel_launch() -> crate::Result<()> {
        let device = CUDA::<Base>::new(0)?;

        let a = Buffer::from((&device, [1, 2, 3, 4, 5]));
        let b = Buffer::from((&device, [4, 1, 7, 6, 9]));

        let mut c = Buffer::<i32, _>::new(&device, a.len());

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
            device.stream(),
            &mut [
                &a.ptr as *const u64 as *mut c_void,
                &b.ptr as *const u64 as *mut c_void,
                &mut c.ptr as *mut u64 as *mut c_void,
                &a.len() as *const usize as *mut c_void,
            ],
        )?;

        assert_eq!(&vec![5, 3, 10, 10, 14], &c.read());
        Ok(())
    }
}

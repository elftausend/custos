pub mod api;
mod cuda_device;
mod kernel_cache;
mod kernel_launch;

pub use cuda_device::*;
pub use kernel_cache::*;
pub use kernel_launch::*;

use crate::{Buffer, CDatatype};

/// Sets the elements of a CUDA Buffer to zero.
/// # Example
/// ```
/// use custos::{CudaDevice, Buffer, VecRead, cuda::cu_clear};
///
/// fn main() -> Result<(), custos::Error> {
///     let device = CudaDevice::new(0)?;
///     let mut lhs = Buffer::<i32>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(device.read(&lhs), vec![15, 30, 21, 5, 8]);
///
///     cu_clear(&device, &mut lhs);
///     assert_eq!(device.read(&lhs), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn cu_clear<T: CDatatype>(device: &CUDA, buf: &mut Buffer<T>) -> crate::Result<()> {
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

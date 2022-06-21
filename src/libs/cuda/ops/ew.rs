use crate::{CDatatype, Buffer, cuda::{CudaCache, launch_kernel1d}, InternCudaDevice};

/// Element-wise operations. The op/operation is usually "+", "-", "*", "/".
/// 
/// # Example
/// ```
/// use custos::{CudaDevice, Buffer, VecRead, cuda::cu_ew};
/// 
/// fn main() -> Result<(), custos::Error> {
///     let device = CudaDevice::new(0)?;
///     let lhs = Buffer::<i32>::from((&device, [15, 30, 21, 5, 8]));
///     let rhs = Buffer::<i32>::from((&device, [10, 9, 8, 6, 3]));
/// 
///     let result = cu_ew(&device, &lhs, &rhs, "+")?;
///     assert_eq!(vec![25, 39, 29, 11, 11], device.read(&result));
///     Ok(())
/// }
/// ```
pub fn cu_ew<T: CDatatype>(device: &InternCudaDevice, lhs: &Buffer<T>, rhs: &Buffer<T>, op: &str) -> crate::Result<Buffer<T>> {
    let src = format!(
        r#"extern "C" __global__ void ew({datatype}* lhs, {datatype}* rhs, {datatype}* out, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    out[idx] = lhs[idx] {op} rhs[idx];
                }}
              
            }}
    "#, datatype=T::as_c_type_str());

    let out: Buffer<T> = CudaCache::get(device, lhs.len);

    launch_kernel1d(
        lhs.len, device, 
        &src, "ew", 
        vec![lhs, rhs, &out, &lhs.len],
    )?;
    
    /*
    let function = fn_cache(device, &src, "ew")?;
    culaunch_kernel(
        &function, [lhs.len as u32, 1, 1], 
        [1, 1, 1], &mut device.stream(), 
        &mut [
            &lhs.ptr.2 as *const u64 as *mut c_void,
            &rhs.ptr.2 as *const u64 as *mut c_void,
            &out.ptr.2 as *const u64 as *mut c_void,
            &lhs.len as *const usize as *mut c_void,
            
        ]
    )?;*/
    Ok(out)
}

/// Element-wise "assign" operations. The op/operation is usually "+", "-", "*", "/".
/// 
/// # Example
/// ```
/// use custos::{CudaDevice, Buffer, VecRead, cuda::cu_ew_self};
/// 
/// fn main() -> Result<(), custos::Error> {
///     let device = CudaDevice::new(0)?;
///     let mut lhs = Buffer::<i32>::from((&device, [15, 30, 21, 5, 8]));
///     let rhs = Buffer::<i32>::from((&device, [10, 9, 8, 6, 3]));
/// 
///     cu_ew_self(&device, &mut lhs, &rhs, "+")?;
///     assert_eq!(vec![25, 39, 29, 11, 11], device.read(&lhs));
///     Ok(())
/// }
/// ```
pub fn cu_ew_self<T: CDatatype>(device: &InternCudaDevice, lhs: &mut Buffer<T>, rhs: &Buffer<T>, op: &str) -> crate::Result<()> {
    let src = format!(
        r#"extern "C" __global__ void ew_self({datatype}* lhs, {datatype}* rhs, int numElements)
            {{  
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    lhs[idx] = lhs[idx] {op} rhs[idx];
                }}
            }}
    "#, datatype=T::as_c_type_str());

    launch_kernel1d(
        lhs.len, device, 
        &src, "ew_self", 
        vec![lhs, rhs, &lhs.len],
    )?;
    Ok(())
}
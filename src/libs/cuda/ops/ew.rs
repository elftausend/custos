use std::ffi::c_void;
use crate::{CDatatype, Buffer, cuda::{fn_cache, api::launch_kernel, CudaCache}, InternCudaDevice};

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

    let out = CudaCache::get(device, lhs.len);

    let function = fn_cache(device, &src, "ew")?;
    launch_kernel(
        &function, [lhs.len as u32, 1, 1], 
        [1, 1, 1], &mut device.stream(), 
        &mut [
            &lhs.ptr.2 as *const u64 as *mut c_void,
            &rhs.ptr.2 as *const u64 as *mut c_void,
            &out.ptr.2 as *const u64 as *mut c_void,
            &lhs.len as *const usize as *mut c_void,
        ]
    )?;
    Ok(out)
}


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

    
    let function = fn_cache(device, &src, "ew_self")?;
    launch_kernel(
        &function, [lhs.len as u32, 1, 1], 
        [1, 1, 1], &mut device.stream(), 
        &mut [
            &lhs.ptr.2 as *const u64 as *mut c_void,
            &rhs.ptr.2 as *const u64 as *mut c_void,
            &lhs.len as *const usize as *mut c_void
        ]
    )?;
    Ok(())
}
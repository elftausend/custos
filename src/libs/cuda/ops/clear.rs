use std::ffi::c_void;

use crate::{CDatatype, cuda::{fn_cache, api::launch_kernel}, Buffer, InternCudaDevice};

pub fn cu_clear<T: CDatatype>(device: &InternCudaDevice, buf: &mut Buffer<T>) -> crate::Result<()> {
    let src = format!(
        r#"extern "C" __global__ void clear({datatype}* self)
            {{
                int idx = blockIdx.x;
                self[idx] = 0;
            }}
    "#, datatype=T::as_c_type_str());

    let function = fn_cache(device, &src, "clear")?;
    launch_kernel(
        &function, [buf.len as u32, 1, 1], 
        [1, 1, 1], &mut device.stream(), 
        &mut [&buf.ptr.2 as *const u64 as *mut c_void]
    )?;
    Ok(())
}
use crate::{CDatatype, cuda::launch_kernel1d, Buffer, InternCudaDevice};

pub fn cu_clear<T: CDatatype>(device: &InternCudaDevice, buf: &mut Buffer<T>) -> crate::Result<()> {
    let src = format!(
        r#"extern "C" __global__ void clear({datatype}* self, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    self[idx] = 0;
                }}
                
            }}
    "#, datatype=T::as_c_type_str());

    launch_kernel1d(
        buf.len, device, 
        &src, "clear", 
        vec![buf, &buf.len],
    )?;

    Ok(())
}
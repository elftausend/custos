use custos::{Buffer, CDatatype, OpenCL, opencl::enqueue_kernel, Device};

use super::ElementWise;

pub fn cl_element_wise<T>(
    device: &OpenCL,
    lhs: &Buffer<T, OpenCL>,
    rhs: &Buffer<T, OpenCL>,
    out: &mut Buffer<T, OpenCL>,
    op: &str,
) -> custos::Result<()> where
    T: CDatatype,
{
    let src = format!(
        "__kernel void cl_ew(__global {datatype}* lhs, __global {datatype}* rhs, __global {datatype} out) {{
            size_t idx = get_global_id(0);

            out[idx] = lhs[idx] {op} rhs[idx];
        }}"
    , datatype=T::as_c_type_str());

    enqueue_kernel(device, &src, [lhs.len, 0, 0], None, 
        &[lhs, rhs, out]
    )?;
    Ok(())
}

impl<T: CDatatype> ElementWise<T, OpenCL> for OpenCL {
    #[inline]
    fn add(&self, lhs: &Buffer<T, OpenCL>, rhs: &Buffer<T, OpenCL>) -> Buffer<T, OpenCL> {
        let mut out = self.retrieve(lhs.len, (lhs, rhs));
        cl_element_wise(self, lhs, rhs, &mut out, "+").unwrap();
        out
    }

    #[inline]
    fn mul(&self, lhs: &Buffer<T, OpenCL>, rhs: &Buffer<T, OpenCL>) -> Buffer<T, OpenCL> {
        let mut out = self.retrieve(lhs.len, (lhs, rhs));
        cl_element_wise(self, lhs, rhs, &mut out, "*").unwrap();
        out
    }
}

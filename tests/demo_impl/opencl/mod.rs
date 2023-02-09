use custos::{opencl::enqueue_kernel, Buffer, CDatatype, Device, OpenCL, Shape};

use super::ElementWise;

pub fn cl_element_wise<T, S: Shape>(
    device: &OpenCL,
    lhs: &Buffer<T, OpenCL, S>,
    rhs: &Buffer<T, OpenCL, S>,
    out: &mut Buffer<T, OpenCL, S>,
    op: &str,
) -> custos::Result<()>
where
    T: CDatatype,
{
    let src = format!(
        "__kernel void cl_ew(__global {datatype}* lhs, __global {datatype}* rhs, __global {datatype}* out) {{
            size_t idx = get_global_id(0);

            out[idx] = lhs[idx] {op} rhs[idx];
        }}"
    , datatype=T::as_c_type_str());

    enqueue_kernel(device, &src, [lhs.len(), 0, 0], None, &[lhs, rhs, out])?;
    Ok(())
}

impl<T: CDatatype, S: Shape> ElementWise<T, OpenCL, S> for OpenCL {
    #[inline]
    fn add(&self, lhs: &Buffer<T, OpenCL, S>, rhs: &Buffer<T, OpenCL, S>) -> Buffer<T, OpenCL, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        cl_element_wise(self, lhs, rhs, &mut out, "+").unwrap();
        out
    }

    #[inline]
    fn mul(&self, lhs: &Buffer<T, OpenCL, S>, rhs: &Buffer<T, OpenCL, S>) -> Buffer<T, OpenCL, S> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        cl_element_wise(self, lhs, rhs, &mut out, "*").unwrap();
        out
    }
}

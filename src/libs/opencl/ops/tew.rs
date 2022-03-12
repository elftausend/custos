use std::fmt::Write;

use crate::{libs::opencl::{GenericOCL, KernelOptions, CLDevice, api::OCLError}, matrix::Matrix};


pub fn tew<T: GenericOCL>(device: CLDevice, lhs: Matrix<T>, rhs: Matrix<T>, op: &str) -> Result<Matrix<T>, OCLError> {
    let mut src = String::new();
    write!(&mut src, r#"
        __kernel void eop(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]{op}rhs[id];
        }}
    "#, datatype=T::as_ocl_type_str(), op=op).unwrap();

    let gws = [lhs.size(), 0, 0];
    KernelOptions::<T>::new(device, lhs, gws, &src)
        .with_rhs(rhs)
        .with_output(lhs.dims())
        .run()
}
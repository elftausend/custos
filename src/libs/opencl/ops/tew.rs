use crate::{libs::opencl::{GenericOCL, KernelOptions, cl_device::InternCLDevice}, matrix::Matrix, Error};

trait Both {
    fn as_str<'a, >() -> &'a str;
}


/*
impl <T: GenericOCL>Both for T {
    fn as_str<'a>() -> &'a str {
        T::as_ocl_type_str()
    }
}


impl <T: !GenericOCL>Both for T {
    fn as_str<'a, >() -> &'a str {
        "undefined"
    }
}
*/

//std::any::TypeId::of::<T>() ... check all impl


pub fn tew<T: GenericOCL>(device: InternCLDevice, lhs: Matrix<T>, rhs: Matrix<T>, op: &str) -> Result<Matrix<T>, Error> {
    let src = format!("
        __kernel void eop(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]{op}rhs[id];
        }}
    ", datatype=T::as_ocl_type_str());

    let gws = [lhs.size(), 0, 0];
    KernelOptions::<T>::new(device, lhs, gws, &src)
        .with_rhs(rhs)
        .with_output(lhs.dims())
        .run()
}

pub fn tew_self<T: GenericOCL>(device: InternCLDevice, lhs: &mut Matrix<T>, rhs: Matrix<T>, op: &str) -> Result<(), Error> {
    let src = format!("
        __kernel void eop_self(__global {datatype}* self, __global const {datatype}* rhs) {{
            size_t id = get_global_id(0);
            self[id] = self[id]{op}rhs[id];
        }}
    ", datatype=T::as_ocl_type_str());

    let gws = [lhs.size(), 0, 0];
    KernelOptions::<T>::new(device, *lhs, gws, &src)
        .with_rhs(rhs)
        .with_output(lhs.dims())
        .run().unwrap();
    Ok(())
}
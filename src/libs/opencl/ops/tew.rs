use crate::{libs::opencl::{KernelOptions, cl_device::InternCLDevice}, Error, GenericOCL, Buffer};

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

/// Elementwise operations. The op/operation is usually "+", "-", "*", "/".
/// 
/// # Example
/// ```
/// use custos::{CLDevice, Buffer, VecRead, opencl::tew};
/// 
/// fn main() -> Result<(), custos::Error> {
///     let device = CLDevice::get(0)?;
///     let lhs = Buffer::<i16>::from((&device, [15, 30, 21, 5, 8]));
///     let rhs = Buffer::<i16>::from((&device, [10, 9, 8, 6, 3]));
/// 
///     let result = tew(&device, &lhs, &rhs, "+")?;
///     assert_eq!(vec![25, 39, 29, 11, 11], device.read(&result));
///     Ok(())
/// }
/// ```
pub fn tew<T: GenericOCL>(device: &InternCLDevice, lhs: &Buffer<T>, rhs: &Buffer<T>, op: &str) -> Result<Buffer<T>, Error> {
    let src = format!("
        __kernel void eop(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]{op}rhs[id];
        }}
    ", datatype=T::as_ocl_type_str());

    let gws = [lhs.len, 0, 0];
    KernelOptions::<T>::new(device, lhs, gws, &src)
        .with_rhs(rhs)
        .with_output(lhs.len)
        .run()
}

pub fn tew_self<T: GenericOCL>(device: &InternCLDevice, lhs: &mut Buffer<T>, rhs: &Buffer<T>, op: &str) -> Result<(), Error> {
    let src = format!("
        __kernel void eop_self(__global {datatype}* self, __global const {datatype}* rhs) {{
            size_t id = get_global_id(0);
            self[id] = self[id]{op}rhs[id];
        }}
    ", datatype=T::as_ocl_type_str());

    let gws = [lhs.len, 0, 0];
    KernelOptions::<T>::new(device, lhs, gws, &src)
        .with_rhs(rhs)
        .with_output(lhs.len)
        .run()?;
    Ok(())
    
}

use crate::{GenericOCL, InternCLDevice, Buffer, opencl::KernelOptions, Error};

/// Sets all the elements of an OpenCL Buffer to zero.
/// # Example
/// ```
/// use custos::{CLDevice, Buffer, VecRead, opencl::cl_clear};
/// 
/// fn main() -> Result<(), custos::Error> {
///     let device = CLDevice::get(0)?;
///     let mut lhs = Buffer::<i16>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(device.read(&lhs), vec![15, 30, 21, 5, 8]);
/// 
///     cl_clear(&device, &mut lhs);
///     assert_eq!(device.read(&lhs), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn cl_clear<T: GenericOCL>(device: &InternCLDevice, lhs: &mut Buffer<T>) -> Result<Buffer<T>, Error> {
    let src = format!("
        __kernel void clear(__global {datatype}* self) {{
            size_t id = get_global_id(0);
            self[id] = 0;
        }}
    ", datatype=T::as_ocl_type_str());

    let gws = [lhs.len, 0, 0];
    KernelOptions::<T>::new(device, lhs, gws, &src)?
        .run()
}
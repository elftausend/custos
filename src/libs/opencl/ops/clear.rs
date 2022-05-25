use crate::{GenericOCL, InternCLDevice, Buffer, opencl::KernelOptions, Error};



pub fn cl_clear<T: GenericOCL>(device: &InternCLDevice, lhs: &mut Buffer<T>) -> Result<Buffer<T>, Error> {
    let src = format!("
        __kernel void clear(__global {datatype}* self) {{
            size_t id = get_global_id(0);
            self[id] = 0;
        }}
    ", datatype=T::as_ocl_type_str());

    let gws = [lhs.len, 0, 0];
    KernelOptions::<T>::new(device, lhs, gws, &src)
        .run()
}
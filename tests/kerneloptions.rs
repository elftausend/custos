use custos::{Error, CLDevice, Buffer, GenericOCL, opencl::KernelOptions, VecRead};


#[test]
fn test_kernel_options() -> Result<(), Error> {
    let device = CLDevice::get(0)?;

    let lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));
    let rhs = Buffer::<i32>::from((&device, [-2, -6, -4, -3, -8, -9]));

    let src = format!("
        __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]+rhs[id];
        }}
    ", datatype=i32::as_ocl_type_str());

    let gws = [lhs.len, 0, 0];
    let out = KernelOptions::<i32>::new(&device, &lhs, gws, &src)?
        .with_rhs(&rhs)
        .with_output(lhs.len)
        .run()?;

    assert_eq!(device.read(&out), vec![-1, -1, -1, -1, -1, -1]);
    Ok(())
}

#[test]
fn test_kernel_options_num_arg() -> Result<(), Error> {
    let device = CLDevice::get(0)?;

    let lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));

    let src = format!("
        __kernel void add_scalar(__global {datatype}* self, {datatype} rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]+rhs;
        }}
    ", datatype=i32::as_ocl_type_str());

    let gws = [lhs.len, 0, 0];
    let out = KernelOptions::<i32>::new(&device, &lhs, gws, &src)?
        .add_arg(&3)
        .with_output(lhs.len)
        .run()?;

    assert_eq!(device.read(&out), vec![4, 8, 6, 5, 10, 11]);
    Ok(())
}
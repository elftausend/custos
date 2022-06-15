
#[cfg(feature="opencl")] 
use custos::{Error, CLDevice, Buffer, CDatatype, opencl::{KernelOptions, KernelRunner}, VecRead};

#[cfg(feature="opencl")] 
#[test]
fn test_kernel_options() -> Result<(), Error> {
    let device = CLDevice::new(0)?;

    let lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));
    let rhs = Buffer::<i32>::from((&device, [-2, -6, -4, -3, -8, -9]));

    let src = format!("
        __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]+rhs[id];
        }}
    ", datatype=i32::as_c_type_str());

    let gws = [lhs.len, 0, 0];
    let out = KernelOptions::<i32>::new(&device, &lhs, gws, &src)?
        .with_rhs(&rhs)
        .with_output(lhs.len)
        .run()?;

    assert_eq!(device.read(&out), vec![-1, -1, -1, -1, -1, -1]);
    Ok(())
}

#[cfg(feature="opencl")] 
#[test]
fn test_kernel_options_num_arg() -> Result<(), Error> {
    let device = CLDevice::new(0)?;

    let mut lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));

    let src = format!("
        __kernel void add_scalar(__global {datatype}* self, float rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]+rhs;
        }}
    ", datatype=i32::as_c_type_str());

    let gws = [lhs.len, 0, 0];
    let out = KernelRunner::<i32>::new(&device, &mut lhs, gws, &src)?
        .add_arg(&mut 3f32)
        .with_output(gws[0])
        .run()?.unwrap();

    assert_eq!(device.read(&out), vec![4, 8, 6, 5, 10, 11]);
    Ok(())
}

#[cfg(feature="opencl")] 
#[test]
fn test_kernel_options_num_arg_assign() -> Result<(), Error> {
    let device = CLDevice::new(0)?;

    let mut lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));

    let src = format!("
        __kernel void add_assign_scalar(__global {datatype}* self, float rhs) {{
            size_t id = get_global_id(0);
            self[id] += rhs;
        }}
    ", datatype=i32::as_c_type_str());

    let gws = [lhs.len, 0, 0];
    KernelRunner::<i32>::new(&device, &mut lhs, gws, &src)?
        .add_arg(&mut 3f32)
        .run()?;

    assert_eq!(device.read(&lhs), vec![4, 8, 6, 5, 10, 11]);
    Ok(())
}
#[cfg(feature="opencl")]
use custos::{opencl::KernelOptions, CLDevice, Error, GenericOCL, VecRead, Buffer};

#[cfg(feature="opencl")]
fn main() -> Result<(), Error> {
    let device = CLDevice::new(0)?;

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

#[cfg(not(feature="opencl"))]
fn main() {}
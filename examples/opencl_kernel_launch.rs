use custos::{cache::Cache, opencl::enqueue_kernel, Buffer, CDatatype, Error, OpenCL};

fn main() -> Result<(), Error> {
    let device = OpenCL::new(0)?;

    let lhs = Buffer::<i32, _>::from((&device, [1, 5, 3, 2, 7, 8]));
    let rhs = Buffer::<i32, _>::from((&device, [-2, -6, -4, -3, -8, -9]));

    let src = format!("
        __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]+rhs[id];
        }}
    ", datatype=i32::as_c_type_str());

    let gws = [lhs.len(), 0, 0];

    let out = Cache::get::<i32, ()>(&device, lhs.len(), ());
    enqueue_kernel(&device, &src, gws, None, &[&lhs, &rhs, &out])?;
    assert_eq!(out.read(), vec![-1, -1, -1, -1, -1, -1]);
    Ok(())
}

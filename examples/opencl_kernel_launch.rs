use custos::{Buffer, CDatatype, CLDevice, Error, opencl::{CLCache, enqueue_kernel}, AsDev};

fn main() -> Result<(), Error> {
    let device = CLDevice::new(0)?.select();

    let lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));
    let rhs = Buffer::<i32>::from((&device, [-2, -6, -4, -3, -8, -9]));

    let src = format!("
        __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]+rhs[id];
        }}
    ", datatype=i32::as_c_type_str());

    let gws = [lhs.len, 0, 0];

    let out = CLCache::get::<i32>(&device, lhs.len);
    enqueue_kernel(&device, &src, gws, None, &[&lhs, &rhs, &out])?;
    assert_eq!(out.read(), vec![-1, -1, -1, -1, -1, -1]);
    Ok(())
}

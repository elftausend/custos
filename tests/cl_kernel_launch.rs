use std::ffi::c_void;

use custos::{opencl::{AsClCvoidPtr, CLCache, enqueue_kernel}, CLDevice, Buffer, AsDev, CDatatype};


#[test]
fn test_as_cl_cvoid() -> custos::Result<()> {
    let x = 5.;
    let ptr = x.as_cvoid_ptr();
    assert_eq!(ptr, &x as *const f64 as *mut c_void);

    let device = CLDevice::new(0)?;
    let buf = Buffer::<f32>::new(&device, 100);
    let ptr = buf.as_cvoid_ptr();
    assert_eq!(ptr, buf.cl_ptr());

    Ok(())
}

#[test]
fn test_kernel_launch() -> custos::Result<()> {
    let device = CLDevice::new(0)?.select();

    let src_add = "
        __kernel void operation(__global const float* lhs, __global float* out, const float add) {
            size_t id = get_global_id(0);
            out[id] = lhs[id] + add;
        }
    ";

    let lhs = Buffer::<f32>::from((&device, [1., 3., 6., 4., 1., 4.,]));
    let out = CLCache::get::<f32>(&device, lhs.len);

    let gws = [lhs.len, 0, 0];
    enqueue_kernel(&device, src_add, gws, None, 
        &[&lhs, &out, &4f32]
    )?;
    assert_eq!(out.read(), vec![5., 7., 10., 8., 5., 8.]);

    Ok(())
}

#[test]
fn test_kernel_launch_diff_datatype() -> custos::Result<()> {
    let device = CLDevice::new(0)?.select();

    let src_add = "
        __kernel void operation(__global const float* lhs, __global float* out, const int add) {
            size_t id = get_global_id(0);
            out[id] = pow(lhs[id], add);
        }
    ";

    let lhs = Buffer::<f32>::from((&device, [1., 3., 6., 4., 1., 4.,]));
    let out = CLCache::get::<f32>(&device, lhs.len);

    let gws = [lhs.len, 0, 0];
    enqueue_kernel(&device, src_add, gws, None, 
        &[&lhs, &out, &3i32]
    )?;
    assert_eq!(out.read(), vec![1., 27., 216., 64., 1., 64.]);

    Ok(())
}

#[test]
fn test_kernel_launch_2() -> custos::Result<()>{
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
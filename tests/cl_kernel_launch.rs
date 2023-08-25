use std::ffi::c_void;

use custos::{
    opencl::{enqueue_kernel, AsClCvoidPtr},
    prelude::{chosen_cl_idx, Float},
    Base, Buffer, CDatatype, OpenCL, Retriever,
};

#[test]
fn test_as_cl_cvoid() -> custos::Result<()> {
    let x = 5.;
    let ptr = x.as_cvoid_ptr();
    assert_eq!(ptr, &x as *const f64 as *mut c_void);

    let device = OpenCL::<Base>::new(chosen_cl_idx())?;
    let buf = Buffer::<f32, _>::new(&device, 100);
    let ptr = buf.as_cvoid_ptr();
    assert_eq!(ptr, buf.cl_ptr());

    Ok(())
}

#[test]
fn test_kernel_launch() -> custos::Result<()> {
    let device = OpenCL::<Base>::new(chosen_cl_idx())?;

    let src_add = "
        __kernel void operation(__global const float* lhs, __global float* out, const float add) {
            size_t id = get_global_id(0);
            out[id] = lhs[id] + add;
        }
    ";

    let lhs = Buffer::<f32, _>::from((&device, [1., 3., 6., 4., 1., 4.]));

    let out: Buffer<'_, f32, OpenCL> = device.retrieve::<(), 0>(lhs.len(), ());

    let gws = [lhs.len(), 0, 0];
    enqueue_kernel(&device, src_add, gws, None, &[&lhs, &out, &4f32])?;
    assert_eq!(out.read(), vec![5., 7., 10., 8., 5., 8.]);

    Ok(())
}

pub fn roughly_eq_slices<T: Float>(lhs: &[T], rhs: &[T]) {
    for (a, b) in lhs.iter().zip(rhs) {
        if (*a - *b).abs() >= T::as_generic(0.1) {
            panic!(
                "Slices 
                left {lhs:?} 
                and right {rhs:?} do not equal. 
                Encountered diffrent value: {a}, {b}"
            )
        }
    }
}

#[test]
fn test_kernel_launch_diff_datatype() -> custos::Result<()> {
    let device = OpenCL::<Base>::new(chosen_cl_idx())?;

    let src_add = "
        __kernel void operation(__global const float* lhs, __global float* out, const int add) {
            size_t id = get_global_id(0);
            out[id] = pow(lhs[id], add);
        }
    ";

    let lhs = Buffer::<f32, _>::from((&device, [1., 3., 6., 4., 1., 4.]));

    let out: Buffer<'_, f32, OpenCL, ()> = device.retrieve::<(), 0>(lhs.len(), ());

    let gws = [lhs.len(), 0, 0];
    enqueue_kernel(&device, src_add, gws, None, &[&lhs, &out, &3i32])?;

    roughly_eq_slices(&out.read(), &[1., 27., 216., 64., 1., 64.]);

    Ok(())
}

#[test]
fn test_kernel_launch_2() -> custos::Result<()> {
    let device = OpenCL::<Base>::new(chosen_cl_idx())?;

    let lhs = Buffer::<i32, _>::from((&device, [1, 5, 3, 2, 7, 8]));
    let rhs = Buffer::<i32, _>::from((&device, [-2, -6, -4, -3, -8, -9]));

    let src = format!("
        __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id]+rhs[id];
        }}
    ", datatype=i32::C_DTYPE_STR);

    let gws = [lhs.len(), 0, 0];

    let out: Buffer<'_, i32, OpenCL> = device.retrieve::<(), 0>(lhs.len(), ());
    enqueue_kernel(&device, &src, gws, None, &[&lhs, &rhs, &out])?;
    assert_eq!(out.read(), vec![-1, -1, -1, -1, -1, -1]);
    Ok(())
}

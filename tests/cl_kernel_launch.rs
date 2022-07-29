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


#[cfg(feature = "opencl")]
#[test]
fn test_enqueue_kernel_num() -> custos::Result<()> {
    use std::ffi::c_void;

    use custos::opencl::{AsClCvoidPtr, CLCache};

    let device = CLDevice::new(0)?.select();

    let lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));

    let src = format!(
        "
        __kernel void add_assign_scalar(__global {i32}* lhs, __global {i32}* out, {i32} rhs) {{
            size_t id = get_global_id(0);
            out[id] += lhs[id] + (int) rhs;
        }}
    ",
        i32 = i32::as_c_type_str()
    );

    let gws = [lhs.len, 0, 0];
    let out = CLCache::get::<i32>(&device, lhs.len);

    // TODO: get this to work......
    //enqueue_kernel(&device, &src, gws, None, vec![&lhs, &out, &3i64])?;
    let kernel = custos::opencl::CL_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .arg_kernel_cache(&device, src.to_string())
    })?;

    let x = 3;
    custos::opencl::api::set_kernel_arg_ptr(
        &kernel,
        0,
        &lhs.as_cvoid_ptr(),
        std::mem::size_of::<*mut c_void>(),
    )?;
    custos::opencl::api::set_kernel_arg_ptr(
        &kernel,
        1,
        &out.as_cvoid_ptr(),
        std::mem::size_of::<*mut c_void>(),
    )?;
    let value = x as *mut c_void;
    custos::opencl::api::set_kernel_arg_ptr(&kernel, 2, &value, 4)?;

    custos::opencl::api::enqueue_nd_range_kernel(&device.queue(), &kernel, 1, &gws, None, None)?;

    assert_eq!(out.read(), vec![4, 8, 6, 5, 10, 11]);
    Ok(())
}

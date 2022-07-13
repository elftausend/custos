use custos::{
    cuda::{launch_kernel1d, CudaCache},
    AsDev, Buffer, CDatatype, CudaDevice, VecRead,
};

#[test]
fn test_scalar_op_cuda() -> custos::Result<()> {
    let device = CudaDevice::new(0)?.select();

    let src = format!(
        r#"extern "C" __global__ void scalar_add({datatype}* lhs, {datatype} rhs, {datatype}* out, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    out[idx] = lhs[idx] + rhs;
                }}
              
            }}
    "#,
        datatype = f32::as_c_type_str()
    );

    let lhs = Buffer::from((&device, [1f32, 2., 3., 4., 5.]));
    let out: Buffer<f32> = CudaCache::get(&device, lhs.len);

    launch_kernel1d(
        lhs.len,
        &device,
        &src,
        "scalar_add",
        vec![&lhs, &3f32, &out, &lhs.len],
    )?;

    assert_eq!(vec![4., 5., 6., 7., 8.], device.read(&out));
    Ok(())
}
